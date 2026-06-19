/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
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
__global__ void ddaAllReduceFlatIpc(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier,
    const T* __restrict__ acc) {
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true  /* hasPreviousMemAccess */,
      true  /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();

  // pattern=2: full reduce into recvbuff (one-shot, not scatter)
  reduceScatter<T, NRANKS, hasAcc>(
      ipcbuffs, recvbuff, acc, selfRank, idxStart, idxEnd, idxStride, 2);

  barrier.syncOnSameBlockIdx<
      true  /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();
}

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceFlatIpcWrite(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier,
    const T* __restrict__ acc) {
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // Remote write: push sendbuff into selfRank's slot of every rank's IPC buffer.
  // Caller must allocate ipcbuffs[r] with capacity NRANKS * count.
  for (int r = 0; r < NRANKS; r++) {
    copyFromSrcToDest<T>(
        sendbuff,
        ipcbuffs[r] + selfRank * count,
        idxStart,
        idxEnd,
        idxStride);
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      true /* prevRemoteWrite */ >();

  // Local reduce: ipcbuffs[selfRank] now holds NRANKS contributions in slots 0..NRANKS-1.
  localReduce<T, NRANKS, hasAcc>(
      ipcbuffs[selfRank], recvbuff, acc, idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */ >();
}

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceTreeIpcWrite(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier,
    const T* __restrict__ acc) {
  const size_t countPerRank = count / NRANKS;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  // Ensure all IPC buffers are ready to receive before any rank starts writing.
  barrier.syncOnSameBlockIdx<
      false,
      true,
      false >();

  // Remote write reduce-scatter: for each target rank r, push the chunk of
  // sendbuff destined for r into selfRank's slot of r's IPC buffer.
  // ipcbuffs[r][s * countPerRank .. (s+1)*countPerRank) = contribution from rank s.
  reduceScatterWrite<T, NRANKS>(
      ipcbuffs, sendbuff, selfRank, idxStart, idxEnd, idxStride);


  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      true >();

  // Local reduce: reduce NRANKS slots from own IPC buffer; store result back
  // into selfRank's slot (safe — each element is read before being overwritten).
  localReduce<T, NRANKS, hasAcc>(
      ipcbuffs[selfRank],
      ipcbuffs[selfRank] + selfRank * countPerRank,
      acc,
      idxStart,
      idxEnd,
      idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      false >();

  // Remote write all-gather: push selfRank's reduced chunk into selfRank's slot
  // of every rank's IPC buffer (reusing the same slot offsets now that the
  // reduce step has consumed the scatter data).
  allGatherWrite<T, NRANKS>(
      ipcbuffs, ipcbuffs[selfRank] + selfRank * countPerRank, selfRank, idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      true >();

  // Assemble recvbuff: own IPC buffer now holds all NRANKS reduced chunks in
  // their respective slots — copy each slot to the correct output position.
  for (int r = 0; r < NRANKS; r++) {
    copyFromSrcToDest<T>(
        ipcbuffs[selfRank] + r * countPerRank,
        recvbuff + r * countPerRank,
        idxStart,
        idxEnd,
        idxStride);
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false >();

} // namespace meta::comms


template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceTreeIpc(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier,
    const T* __restrict__ acc) {
  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true  /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();

  const size_t countPerRank = count / NRANKS;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  reduceScatter<T, NRANKS, hasAcc>(
      ipcbuffs,
      ipcbuffs[selfRank],
      acc,
      selfRank,
      idxStart,
      idxEnd,
      idxStride,
      1);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();

  allGather<T, NRANKS>(
      ipcbuffs, recvbuff, selfRank, idxStart, idxEnd, idxStride, true);

  barrier.syncOnSameBlockIdx<
      true  /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();
}

} // namespace meta::comms
