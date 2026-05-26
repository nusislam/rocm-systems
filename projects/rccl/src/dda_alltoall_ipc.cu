/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_alltoall_ipc.h"

#include "algorithms/CollCommon.h"
#include "algorithms/alltoall/alltoall_dda.h"
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

/** Flat below this size; Bruck above for logarithmic efficiency. */
constexpr size_t kDdaFlatBruckThresholdBytes = 1ULL << 16; // 64KB

/** Pairwise below this rank count; recursive above for scalability. */
constexpr int kDdaPairwiseRecursiveRankThreshold = 8;

/** Scatter-gather above this size for better memory utilization. */
constexpr size_t kDdaScatterGatherThresholdBytes = 1ULL << 18; // 256KB

/** Large message threshold for advanced algorithms. */
constexpr size_t kDdaLargeMessageThresholdBytes = 1ULL << 20; // 1MB

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
static ncclResult_t ncclAllToAllDdaIpcTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const size_t totalCount = count * comm->nRanks;
  if (totalCount * sizeof(T) > comm->ddaIpcScratchBytes) {
    WARN(
        "DDA IPC alltoall: total element count %zu needs %zu bytes; comm scratch is %zu bytes",
        totalCount,
        totalCount * sizeof(T),
        comm->ddaIpcScratchBytes);
    return ncclInvalidArgument;
  }

  const size_t messageSizeBytes = count * sizeof(T);
  const size_t totalSizeBytes = totalCount * sizeof(T);
  const unsigned threads = 512;

  // Algorithm selection logic based on message size and rank count
  /*const bool wantBruck = totalSizeBytes > kDdaFlatBruckThresholdBytes;
  const bool bruckOk = wantBruck && (count % 16 == 0); // Alignment requirement for Bruck
  const bool wantRecursive = comm->nRanks >= kDdaPairwiseRecursiveRankThreshold;
  const bool wantScatterGather = totalSizeBytes > kDdaScatterGatherThresholdBytes && comm->nRanks >= 4;

  if (wantBruck && !bruckOk) {
    INFO(
        NCCL_ALL,
        "DDA IPC alltoall: size %zu B > 64KB but count %zu not aligned to 16; using flat kernel",
        totalSizeBytes,
        count);
  }*/

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  // For alltoall, we use count for grid calculation (data per rank pair)
  auto gridBlock = getGridAndBlockDims(count, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaIpcPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  CUDACHECK(cudaMemcpyAsync(
        comm->ddaIpcScratch,
        sendbuff,
        totalCount * sizeof(T),
        cudaMemcpyDeviceToDevice,
        stream));

  meta::comms::ddaAllToAllIpc<T, kDdaNranks, false>
      <<<grid, block, 0, stream>>>(
          d_ipcbuffs,
          static_cast<T*>(recvbuff),
          count,
          static_cast<const T*>(sendbuff),
          comm->rank,
          barrierHost);
  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllToAllDdaIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != nccl_dda_ipc_detail::kDdaNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }

  size_t totalCount = count * comm->nRanks;
  size_t need = totalCount * 4;
  if (datatype == ncclFloat16 || datatype == ncclBfloat16) {
    need = totalCount * 2;
  }
  if (need > comm->ddaIpcScratchBytes) {
    return false;
  }

  // Check 16-byte alignment for data
  if ((count * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllToAllDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm* comm,
    cudaStream_t stream) {
  switch (datatype) {
  case ncclFloat32:
    return ncclAllToAllDdaIpcTyped<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllToAllDdaIpcTyped<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllToAllDdaIpcTyped<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}

