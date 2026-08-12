/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher for GIN tree all-reduce (remote writes via gin.put).
 ************************************************************************/

#include "gin_all_reduce.h"

#if defined(ENABLE_ROCSHMEM_GIN) && (defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__))

#include "algorithms/all_reduce/all_reduce_gin_tree_sdma.h"
#include "algorithms/CollCommon.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "dda_init_detail.h"
#include "debug.h"
#include "dev_runtime.h"
#include "gin/gin_host.h"
#include "nccl_device/net_device.h"
#include "sym_kernels.h"

#include <cuda_runtime.h>

namespace {

using nccl_dda_detail::ddaMaxNBlocksForScratch;

constexpr size_t kGinFlatTreeThresholdBytes = 1ULL << 18;
constexpr size_t kGinAllReduceScratchBytes = 128ULL * 1024 * 1024;

static bool ginAllReduceScratchWinReady(ncclComm* comm) {
  return comm != nullptr && comm->ginAllReduceScratchWin != nullptr &&
         comm->ginAllReduceScratchWin->vidmem != nullptr;
}

static ncclResult_t ginAllReduceEnsureScratch(ncclComm* comm) {
  if (ginAllReduceScratchWinReady(comm)) {
    return ncclSuccess;
  }

  void* scratch = nullptr;
  ncclWindow_t scratchWinDev = nullptr;
  NCCLCHECK(ncclMemAlloc(&scratch, kGinAllReduceScratchBytes));
  // Register for GIN/LSA visibility without NCCL_WIN_COLL_SYMMETRIC, which would
  // trigger symk init and interfere with user symmetric window registration (-R 2).
  NCCLCHECK(ncclDevrWindowRegisterInGroup(comm, scratch, kGinAllReduceScratchBytes, /*winFlags=*/0, &scratchWinDev));
  NCCLCHECK(ncclDevrFindWindow(comm, scratch, &comm->ginAllReduceScratchWin));
  if (!ginAllReduceScratchWinReady(comm)) {
    NCCLCHECK(ncclMemFree(scratch));
    return ncclInternalError;
  }
  comm->ginAllReduceScratch = scratch;
  comm->ginAllReduceScratchBytes = kGinAllReduceScratchBytes;
  INFO(NCCL_INIT, "GIN all-reduce scratch allocated: %zu bytes", kGinAllReduceScratchBytes);
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceGinTreeTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                              cudaStream_t stream, struct ncclDevrWindow* sendWin,
                                              struct ncclDevrWindow* recvWin) {
  NCCLCHECK(ncclGinAllReduceInitOnce(comm));
  if (!ginAllReduceScratchWinReady(comm)) {
    return ncclInternalError;
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto gridBlock = meta::comms::getGridAndBlockDims(count, sizeof(T), static_cast<size_t>(nBlocksMax));
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;

  const size_t sendOff = static_cast<size_t>(static_cast<const char*>(sendbuff) - static_cast<const char*>(sendWin->userPtr));
  const size_t recvOff = static_cast<size_t>(static_cast<char*>(recvbuff) - static_cast<char*>(recvWin->userPtr));

  meta::comms::ginAllReduceTreeKernel<T><<<grid, block, 0, stream>>>(
    comm->ginAllReduceDevComm, sendWin->vidmem, sendOff, comm->ginAllReduceScratchWin->vidmem, /*scratchOff=*/0,
    recvWin->vidmem, recvOff, count, comm->nRanks, comm->rank, /*rsSignalBase=*/0,
    /*agSignalBase=*/static_cast<unsigned>(nBlocksMax));

  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

} // namespace

bool ncclGinAllReduceBackendConfigured(ncclComm* comm) {
  if (comm == nullptr || comm->sharedRes == nullptr) {
    return false;
  }
  if (comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE || !comm->symmetricSupport) {
    return false;
  }
  return comm->sharedRes->ginState.ginType == NCCL_NET_DEVICE_GIN_ANVIL_SDMA;
}

bool ncclGinAllReduceSdmaBackendEnabled(ncclComm* comm) {
  if (comm == nullptr || comm->sharedRes == nullptr) {
    return false;
  }
  if (comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE) {
    return false;
  }
  return comm->sharedRes->ginState.connected &&
         comm->sharedRes->ginState.ginType == NCCL_NET_DEVICE_GIN_ANVIL_SDMA;
}

ncclResult_t ncclGinAllReduceScratchInit(ncclComm* comm) {
  if (ginAllReduceScratchWinReady(comm)) {
    return ncclSuccess;
  }
  NCCLCHECK(ncclDevrInitOnce(comm));
  NCCLCHECK(ginAllReduceEnsureScratch(comm));
  return ncclSuccess;
}

ncclResult_t ncclGinAllReduceInitOnce(ncclComm* comm) {
  NCCLCHECK(ncclGinAllReduceScratchInit(comm));

  if (comm->ginAllReduceDevCommReady) {
    return ncclSuccess;
  }

  NCCLCHECK(ncclGinConnectOnce(comm));

  const int nBlocks = ddaMaxNBlocksForScratch();
  struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.railGinBarrierCount = nBlocks;
  reqs.ginSignalCount = 2 * nBlocks;
  reqs.ginConnectionType =
    comm->globalGinSupport == NCCL_GIN_CONNECTION_FULL ? NCCL_GIN_CONNECTION_FULL : NCCL_GIN_CONNECTION_RAIL;

  NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &comm->ginAllReduceDevComm, /*isInternal=*/true));
  comm->ginAllReduceDevCommReady = true;
  return ncclSuccess;
}

ncclResult_t ncclGinAllReduceFinalize(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }
  if (comm->ginAllReduceDevCommReady) {
    NCCLCHECK(ncclDevCommDestroy(comm, &comm->ginAllReduceDevComm));
    comm->ginAllReduceDevCommReady = false;
  }
  if (comm->ginAllReduceScratchWin != nullptr && comm->ginAllReduceScratchWin->vidmem != nullptr) {
    NCCLCHECK(ncclCommWindowDeregister(comm, comm->ginAllReduceScratchWin->vidmem));
    comm->ginAllReduceScratchWin = nullptr;
  }
  if (comm->ginAllReduceScratch != nullptr) {
    NCCLCHECK(ncclMemFree(comm->ginAllReduceScratch));
    comm->ginAllReduceScratch = nullptr;
    comm->ginAllReduceScratchBytes = 0;
  }
  return ncclSuccess;
}

bool ncclAllReduceGinTreeEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr) {
    return false;
  }
  if (!ncclGinAllReduceSdmaBackendEnabled(comm)) {
    return false;
  }
  if (!ginAllReduceScratchWinReady(comm)) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }
  if (count == 0) {
    return false;
  }

  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  ncclDevrFindWindow(comm, sendbuff, &sendWin);
  ncclDevrFindWindow(comm, recvbuff, &recvWin);
  if (sendWin == nullptr || recvWin == nullptr) {
    return false;
  }
  if (!(sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) || !(recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes > comm->ginAllReduceScratchBytes) {
    return false;
  }
  if (bytes % 16) {
    return false;
  }
  if (bytes <= kGinFlatTreeThresholdBytes) {
    return false;
  }
  if (count % static_cast<size_t>(comm->nRanks) != 0) {
    return false;
  }
  if (((count / comm->nRanks) * ncclTypeSize(datatype)) % 16) {
    return false;
  }
  return true;
}

ncclResult_t ncclAllReduceGinTree(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  if (!ncclAllReduceGinTreeEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    return ncclInvalidUsage;
  }

  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  NCCLCHECK(ncclDevrFindWindow(comm, sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, recvbuff, &recvWin));
  if (sendWin == nullptr || recvWin == nullptr) {
    return ncclInvalidUsage;
  }

  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceGinTreeTyped<float>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  case ncclFloat16:
    return ncclAllReduceGinTreeTyped<half>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  case ncclBfloat16:
    return ncclAllReduceGinTreeTyped<bf16>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  default:
    return ncclInvalidArgument;
  }
}

#else

bool ncclGinAllReduceBackendConfigured(ncclComm* comm) {
  (void)comm;
  return false;
}

bool ncclGinAllReduceSdmaBackendEnabled(ncclComm* comm) {
  (void)comm;
  return false;
}

bool ncclAllReduceGinTreeEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op) {
  (void)comm;
  (void)sendbuff;
  (void)recvbuff;
  (void)count;
  (void)datatype;
  (void)op;
  return false;
}

ncclResult_t ncclAllReduceGinTree(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)sendbuff;
  (void)recvbuff;
  (void)count;
  (void)datatype;
  (void)op;
  (void)comm;
  (void)stream;
  return ncclInvalidUsage;
}

ncclResult_t ncclGinAllReduceScratchInit(ncclComm* comm) {
  (void)comm;
  return ncclInvalidUsage;
}

ncclResult_t ncclGinAllReduceInitOnce(ncclComm* comm) {
  (void)comm;
  return ncclInvalidUsage;
}

ncclResult_t ncclGinAllReduceFinalize(ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

#endif
