/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * GIN-SDMA AllReduce for single-node (scaleup-only) symmetric windows.
 * Messages <= 16 MiB use the LSA one-shot kernel; larger messages use LSA two-shot.
 *
 * Compiled with NCCL_GIN_ANVIL_SDMA_ENABLE=1 and NCCL_GIN_PROXY_ENABLE=0 so
 * ncclGinCallImpl resolves the SDMA backend at compile time.
 * See LICENSE.txt for license information.
 ******************************************************************************/

#include "gin_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_gin_sdma.h"
#include "algorithms/CollCommon.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"
#include "param.h"

#include <cuda_runtime.h>

NCCL_PARAM(GinAllReduceEnable, "GIN_ALLREDUCE_ENABLE", 1);

namespace {

constexpr bool kSdmaDeviceBackendCompiled = (NCCL_GIN_ANVIL_SDMA_ENABLE != 0);
constexpr int kGinAllReduceMaxRanks = 16;

static ncclResult_t ncclGinAllReduceInitOnce(ncclComm* comm) {
  NCCLCHECK(ncclDevrInitOnce(comm));
  struct ncclGinAllReduceState* state = &comm->ginAllReduceState;
  if (!state->initialized) {
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = kGinAllReduceLsaCtas;
    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &state->devComm, /*isInternal=*/true));
    state->initialized = true;
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinAllReduceEnsureTwoShotSync(ncclComm* comm) {
  struct ncclGinAllReduceState* state = &comm->ginAllReduceState;
  if (state->reduceDoneSync != nullptr) {
    return ncclSuccess;
  }
  CUDACHECK(cudaMalloc(&state->reduceDoneSync, sizeof(uint64_t)));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceGinSdmaOneShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                     ncclComm* comm, cudaStream_t stream,
                                                     struct ncclDevrWindow* sendWin,
                                                     struct ncclDevrWindow* recvWin) {
  NCCLCHECK(ncclGinAllReduceInitOnce(comm));

  const size_t sendOff =
    static_cast<size_t>(static_cast<const char*>(sendbuff) - static_cast<const char*>(sendWin->userPtr));
  const size_t recvOff =
    static_cast<size_t>(static_cast<char*>(recvbuff) - static_cast<const char*>(recvWin->userPtr));

  meta::comms::allReduceLsaOneShotKernel<T><<<kGinAllReduceLsaCtas, kGinAllReduceLsaThreadsPerCta, 0, stream>>>(
    sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, count, comm->ginAllReduceState.devComm);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceGinSdmaTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                     ncclComm* comm, cudaStream_t stream,
                                                     struct ncclDevrWindow* sendWin,
                                                     struct ncclDevrWindow* recvWin) {
  NCCLCHECK(ncclGinAllReduceInitOnce(comm));
  NCCLCHECK(ncclGinAllReduceEnsureTwoShotSync(comm));

  const size_t sendOff =
    static_cast<size_t>(static_cast<const char*>(sendbuff) - static_cast<const char*>(sendWin->userPtr));
  const size_t recvOff =
    static_cast<size_t>(static_cast<char*>(recvbuff) - static_cast<const char*>(recvWin->userPtr));
  const size_t countPerRank = count / static_cast<size_t>(comm->nRanks);

  CUDACHECK(cudaMemsetAsync(comm->ginAllReduceState.reduceDoneSync, 0, sizeof(uint64_t), stream));

  printf("Two-shot\n");
  meta::comms::lsaAllReduceTwoShotKernel<T><<<kGinAllReduceLsaCtas, kGinAllReduceLsaThreadsPerCta, 0, stream>>>(
    comm->ginAllReduceState.devComm, sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, countPerRank,
    comm->ginAllReduceState.reduceDoneSync, comm->nRanks);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceGinSdmaTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                              cudaStream_t stream, struct ncclDevrWindow* sendWin,
                                              struct ncclDevrWindow* recvWin) {
  const size_t bytes = count * sizeof(T);
  if (bytes < kGinAllReduceLsaOneShotMaxBytes) {
    return ncclAllReduceGinSdmaOneShotTyped<T>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  }
  return ncclAllReduceGinSdmaTwoShotTyped<T>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
}

static bool ginAllReduceTwoShotEligible(size_t count, ncclDataType_t datatype, int nRanks) {
  if (count % static_cast<size_t>(nRanks) != 0) {
    return false;
  }
  const size_t countPerRank = count / static_cast<size_t>(nRanks);
  const size_t typeSize = ncclTypeSize(datatype);
  if ((countPerRank * typeSize) % 16 != 0) {
    return false;
  }
  return true;
}

} // namespace

bool ncclAllReduceGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op) {
  if (!kSdmaDeviceBackendCompiled) return false;
  if (!ncclParamGinAllReduceEnable()) return false;
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr) return false;
  if (count == 0) return false;
  if (op != ncclSum) return false;
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) return false;
  if (!comm->symmetricSupport) return false;
  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) return false;
  if (comm->nNodes != 1) return false;
  if (ncclTeamLsa(comm).nRanks != comm->nRanks) return false;
  if (comm->nRanks > kGinAllReduceMaxRanks) return false;
  if (comm->sharedRes->ginState.ginType != (ncclGinType_t)NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return false;

  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  if (ncclDevrFindWindow(comm, sendbuff, &sendWin) != ncclSuccess || sendWin == nullptr) return false;
  if (ncclDevrFindWindow(comm, recvbuff, &recvWin) != ncclSuccess || recvWin == nullptr) return false;
  if (!(sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) || !(recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes <= kGinAllReduceLsaOneShotMaxBytes) {
    return true;
  }
  return ginAllReduceTwoShotEligible(count, datatype, comm->nRanks);
}

ncclResult_t ncclAllReduceGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  if (!ncclAllReduceGinSdmaEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    return ncclInvalidUsage;
  }

  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  NCCLCHECK(ncclDevrFindWindow(comm, sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, recvbuff, &recvWin));
  if (sendWin == nullptr || recvWin == nullptr) return ncclInvalidUsage;

  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceGinSdmaTyped<float>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  case ncclFloat16:
    return ncclAllReduceGinSdmaTyped<half>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  case ncclBfloat16:
    return ncclAllReduceGinSdmaTyped<bf16>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  default:
    return ncclInvalidArgument;
  }
}

ncclResult_t ncclGinAllReduceFinalize(ncclComm* comm) {
  struct ncclGinAllReduceState* state = &comm->ginAllReduceState;
  if (state->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &state->devComm));
    state->initialized = false;
  }
  if (state->reduceDoneSync != nullptr) {
    CUDACHECK(cudaFree(state->reduceDoneSync));
    state->reduceDoneSync = nullptr;
  }
  return ncclSuccess;
}
