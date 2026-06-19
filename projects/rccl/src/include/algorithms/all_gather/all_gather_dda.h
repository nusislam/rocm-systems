/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
 * Adapted for allgather collective operation.
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "ipc_gpu_barrier.h"
#include "algorithms/CollCommon.h"

namespace meta::comms {

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllGatherIpc(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier) {

  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // It is expensive to launch hipMemcpyAsync on ROCm
  // Move data copy here. Each block copies part of sendbuff data
  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();

  allGather<T, NRANKS>(
      ipcbuffs, recvbuff, selfRank, idxStart, idxEnd, idxStride, false);

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();
}

// Remote-write variant: each rank pushes its chunk into every rank's ipcbuff,
// then each rank does a local copy from its ipcbuff to recvbuff.
// ipcbuffs[r] must be allocated as NRANKS * count elements (not count).
template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllGatherIpcWrite(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier) {

  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // Phase 1: push sendbuff into ipcbuffs[r][selfRank*count .. (selfRank+1)*count)
  // for every rank r (remote write).
  allGatherWrite<T, NRANKS>(
      ipcbuffs, sendbuff, selfRank, idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true  /* hasPreviousMemAccess */,
      true  /* hasSubsequentMemAccess */,
      true  /* prevRemoteWrite */>();

  // Phase 2: local copy — ipcbuffs[selfRank] is now fully populated by all ranks.
  copyFromSrcToDest<T>(
      ipcbuffs[selfRank], recvbuff, idxStart, count * NRANKS, idxStride);

  // Barrier so no rank frees its ipcbuff before peers finish their local copies.
  barrier.syncOnSameBlockIdx<
      true  /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();
}

} // namespace meta::comms
