/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA all-reduce kernels for the fabric/VMM path, using FabricGpuBarrier.
 *
 * The kernels are templated on a compile-time rank count NRANKS_CT:
 *   - NRANKS_CT > 0  : specialized for that clique size; the unified CollCommon
 *                      reduceScatter/allGather fully unroll the peer loop
 *                      (matching the IPC fast path). The host launcher
 *                      instantiates this for the common sizes (e.g. 4, 8).
 *   - NRANKS_CT == 0 : runtime fallback; the rank count is passed via the nRanks
 *                      argument and the unified helpers partially unroll 8-wide,
 *                      so a single instantiation covers any other clique size
 *                      up to kDdaMaxNranks.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/CollCommon.h"
#include "fabric_gpu_barrier.h"

namespace meta::comms {

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceFlatFabric(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanks,
    FabricGpuBarrier barrier,
    const T* __restrict__ acc) {
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();

  // pattern=2: full reduce into recvbuff (one-shot). The unified helper folds
  // nRanks to NRANKS_CT (full unroll) when specialized, else uses the runtime
  // nRanks with an 8-wide partial unroll.
  reduceScatter<T, NRANKS_CT, hasAcc>(
      ipcbuffs, recvbuff, acc, selfRank, nRanks, idxStart, idxEnd, idxStride, 2);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */ >();
}

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceFlatFabricWrite(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanksRuntime,
    FabricGpuBarrier barrier,
    const T* __restrict__ acc) {
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;	
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // Remote write: push sendbuff into selfRank's slot of every rank's IPC buffer.
  // Caller must allocate ipcbuffs[r] with capacity NRANKS * count.
  for (int r = 0; r < nRanks; r++) {
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
      true /* prevRemoteWrite */>();

  // Local reduce: ipcbuffs[selfRank] now holds NRANKS contributions in slots 0..NRANKS-1.
  localReduce<T, NRANKS_CT, hasAcc>(
      ipcbuffs[selfRank], recvbuff, acc, nRanks, idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /*prevRemoteWrite */>();
}

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceTreeFabric(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanks,
    FabricGpuBarrier barrier,
    const T* __restrict__ acc) {
  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      false >();

  // Use the compile-time rank count as the divisor when specialized.
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  const size_t countPerRank = count / nRanksEff;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  // Two-shot: reduce-scatter this rank's shard, then all-gather. The unified
  // helpers fold nRanks to NRANKS_CT (full unroll) when specialized, else use
  // the runtime nRanks with an 8-wide partial unroll.
  reduceScatter<T, NRANKS_CT, hasAcc>(
      ipcbuffs, ipcbuffs[selfRank], acc, selfRank, nRanks, idxStart, idxEnd,
      idxStride, 1);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      false >();

  allGather<T, NRANKS_CT>(
      ipcbuffs, recvbuff, selfRank, nRanks, idxStart, idxEnd, idxStride, true);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false >();
}

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceTreeFabricWrite(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanks,
    FabricGpuBarrier barrier,
    const T* __restrict__ acc) {
  
  // Use the compile-time rank count as the divisor when specialized.
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  const size_t countPerRank = count / nRanksEff;
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
  reduceScatterWrite<T, NRANKS_CT>(
      ipcbuffs, sendbuff, selfRank, nRanks, idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      true >();

  // Local reduce: reduce NRANKS slots from own IPC buffer; store result back
  // into selfRank's slot (safe — each element is read before being overwritten).
  localReduce<T, NRANKS_CT, hasAcc>(
      ipcbuffs[selfRank],
      ipcbuffs[selfRank] + selfRank * countPerRank,
      acc,
      nRanks,
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
  allGatherWrite<T, NRANKS_CT>(
      ipcbuffs, ipcbuffs[selfRank] + selfRank * countPerRank, selfRank, nRanks, idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      true >();

   // Assemble recvbuff: own IPC buffer now holds all NRANKS reduced chunks in
  // their respective slots — copy each slot to the correct output position.
  for (int r = 0; r < NRANKS_CT; r++) {
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
}

} // namespace meta::comms
