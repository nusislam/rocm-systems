/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/reduce_scatter/reduce_scatter_dda.cuh.
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
__global__ void ddaReduceScatterIpc(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier) {

  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true  /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();

  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  reduceScatter<T, NRANKS, hasAcc>(
      ipcbuffs, recvbuff, nullptr, selfRank, idxStart, idxEnd, idxStride, 0);

  barrier.syncOnSameBlockIdx<
      true  /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();
}

// Remote-write variant: each rank scatters its sendbuff slices into remote ipcbuffs,
// then performs a fully local reduction from its own ipcbuff into recvbuff.
// ipcbuffs[r] must be allocated as NRANKS * count elements (not count).
template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaReduceScatterIpcWrite(
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

  // Phase 1: scatter sendbuff[r*count + idx] → ipcbuffs[r][selfRank*count + idx]
  // for all ranks r (remote write, no intermediate staging kernel needed).
  reduceScatterWrite<T, NRANKS>(
      ipcbuffs, sendbuff, selfRank, idxStart, idxEnd, idxStride);

  // Wait for all ranks to finish writing before any rank reads its ipcbuff.
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      true /* prevRemoteWrite */>();

  // Phase 2: reduce ipcbuffs[selfRank][r*count + idx] across all r → recvbuff[idx].
  // All reads are local — ipcbuffs[selfRank] resides on this GPU.
  localReduce<T, NRANKS, hasAcc>(
      ipcbuffs[selfRank], recvbuff, nullptr, idxStart, idxEnd, idxStride);

  // Wait for all ranks to finish reading before ipcbuffs can be released.
  barrier.syncOnSameBlockIdx<
      true  /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false >();
}


} // namespace meta::comms
