/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_reduce_scatter_ipc.h"

#include "algorithms/CollCommon.h"
#include "algorithms/reduce_scatter/reduce_scatter_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ipc_gpu_barrier.h"
#include "ipc_init_detail.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

namespace {

using nccl_dda_ipc_detail::DdaIpcBarrierState;
using nccl_dda_ipc_detail::ddaMaxNBlocksForScratch;
using nccl_dda_ipc_detail::kDdaNranks;

/** Flat below this size; tree above (see ddaReduceScatterFlatIpc / ddaReduceScatterTreeIpc). */
constexpr size_t kernelCopyThresholdBytes = 1ULL << 18;


inline uint32_t divRoundUp(size_t a, size_t b) {
  uint32_t y = static_cast<uint32_t>((a + b - 1) / b);
  if (y == 0) {
    y = 1;
  }
  return y;
}

constexpr uint32_t
calcBlockCount(size_t numThreads, size_t threadsPerBlock, size_t maxBlocks) {
  const auto uNumThreads = static_cast<uint64_t>(numThreads);
  const auto uThreadsPerBlock = static_cast<uint64_t>(threadsPerBlock);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks =
      uNumThreads / uThreadsPerBlock + (uNumThreads % uThreadsPerBlock != 0);
  uint32_t y = static_cast<uint32_t>(std::min(blocks, maxBlocks));
  if (y == 0) {
    y = 1;
  }
  return y;
}

std::pair<dim3, dim3>
getGridAndBlockDims(size_t count, int typeSize, size_t maxBlocks) {
  constexpr uint32_t kThreadsPerWarp = 64;
  constexpr uint32_t kThreadsPerBlock = 512;

  const uint32_t elementsPerThread =
      16 / typeSize; // we do 16 Byte load in kernel

  const uint32_t elementsPerWarp = elementsPerThread * kThreadsPerWarp;

  dim3 threads(0, 1, 1);
  dim3 blocks(0, 1, 1);
  if (count < elementsPerThread * kThreadsPerBlock) {
    threads.x = divRoundUp(count, elementsPerWarp) * kThreadsPerWarp;
    blocks.x = 1;
  } else {
    auto warpsRequired = divRoundUp(count, elementsPerWarp);
    blocks.x = calcBlockCount(
        divRoundUp(count, elementsPerThread), kThreadsPerBlock, maxBlocks);
    auto warpsPerBlock = divRoundUp(warpsRequired, blocks.x);
    auto threadsPerBlock =
        std::min<uint32_t>(kThreadsPerBlock, warpsPerBlock * kThreadsPerWarp);
    threads.x = threadsPerBlock;
  }

  return std::make_pair(blocks, threads);
}

template <typename T>
static ncclResult_t ncclReduceScatterDdaIpcTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclComm* comm,
    cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const size_t totalCount = recvcount * comm->nRanks;
  if (totalCount * sizeof(T) > comm->ddaIpcScratchBytes) {
    WARN(
        "DDA IPC reduce-scatter: total element count %zu needs %zu bytes; comm scratch is %zu bytes",
        totalCount,
        totalCount * sizeof(T),
        comm->ddaIpcScratchBytes);
    return ncclInvalidArgument;
  }

  const size_t totalSizeBytes = totalCount * sizeof(T);
  const unsigned threads = 512;
  /*const bool wantTree = totalSizeBytes > kDdaFlatTreeThresholdBytes;
  const bool treeOk = wantTree && (totalCount % static_cast<size_t>(kDdaNranks) == 0);
  const bool wantRecursive = comm->nRanks >= kDdaRecursiveRankThreshold;

  if (wantTree && !treeOk) {
    INFO(
        NCCL_ALL,
        "DDA IPC reduce-scatter: size %zu B > 256KB but count %zu not divisible by %d; using flat kernel",
        totalSizeBytes,
        totalCount,
        kDdaNranks);
  }*/

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  // For reduce-scatter, we use recvcount for grid calculation since each rank processes its portion
  auto gridBlock = getGridAndBlockDims(recvcount, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaIpcPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  /*if (wantRecursive) {
    meta::comms::ddaReduceScatterRecursiveIpc<T, kDdaNranks, false>
        <<<grid, block, 0, stream>>>(
            d_ipcbuffs,
            static_cast<T*>(recvbuff),
            totalCount,
            static_cast<const T*>(sendbuff),
            comm->rank,
            barrierHost,
            nullptr);
  } else if (treeOk) {
    meta::comms::ddaReduceScatterTreeIpc<T, kDdaNranks, false>
        <<<grid, block, 0, stream>>>(
            d_ipcbuffs,
            static_cast<T*>(recvbuff),
            totalCount,
            static_cast<const T*>(sendbuff),
            comm->rank,
            barrierHost,
            nullptr);
  } else {*/
    // Use flat algorithm for smaller messages
  //if (totalCount * sizeof(T) > kernelCopyThresholdBytes) {
  CUDACHECK(cudaMemcpyAsync(
        comm->ddaIpcScratch,
        sendbuff,
        totalCount * sizeof(T),
        cudaMemcpyDeviceToDevice,
        stream));
  //} 
  meta::comms::ddaReduceScatterIpc<T, kDdaNranks, false>
      <<<grid, block, 0, stream>>>(
          d_ipcbuffs,
          static_cast<T*>(recvbuff),
          recvcount,
          static_cast<const T*>(sendbuff),
          comm->rank,
          barrierHost);
  //}

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclReduceScatterDdaIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (recvcount == 0) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != nccl_dda_ipc_detail::kDdaNranks) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }

  size_t totalCount = recvcount * comm->nRanks;
  size_t need = totalCount * 4;
  if (datatype == ncclFloat16 || datatype == ncclBfloat16) {
    need = totalCount * 2;
  }
  if (need > comm->ddaIpcScratchBytes) {
    return false;
  }

  // Check 16-byte alignment for total data
  if ((totalCount * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  // Check per-rank alignment for tree algorithm
  if ((totalCount * ncclTypeSize(datatype)) > kernelCopyThresholdBytes) {
    if (recvcount % 1 || ((recvcount * ncclTypeSize(datatype)) % 16)) {
      return false;
    }
  }

  return true;
}

ncclResult_t ncclReduceScatterDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclReduceScatterDdaIpcTyped<float>(
        sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat16:
    return ncclReduceScatterDdaIpcTyped<half>(
        sendbuff, recvbuff, recvcount, comm, stream);
  case ncclBfloat16:
    return ncclReduceScatterDdaIpcTyped<bf16>(
        sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}

