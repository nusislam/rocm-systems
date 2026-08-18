/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LSA AllReduce device kernels:
 *   allReduceLsaOneShotKernel — read all peers, sum, write all peers (small messages).
 *   lsaAllReduceTwoShotKernel — LSA reduce-scatter + LSA all-gather into recv.
 *   ginAllReduceTwoShotKernel  — LSA reduce-scatter + single-CTA GIN all-gather from recv.
 *
 * One-shot follows projects/rccl-tests/src/all_reduce.cu allReduceLsaKernel.
 ******************************************************************************/

#pragma once

#include "nccl_device.h"
#include "nccl_device/ptr.h"
#include "algorithms/CollCommon.h"

namespace meta::comms {

template <typename T>
__device__ __forceinline__ T allReduceLsaSumAdd(T a, T b) {
  if constexpr (std::is_same<T, bf16>::value) {
    return __hadd(a, b);
  } else {
    return a + b;
  }
}

__device__ __forceinline__ void allReduceTwoShotCtaArrive(uint64_t* counter, uint64_t target, ncclCoopCta cta) {
  cta.sync();
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(counter, 1ULL, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  cta.sync();
  while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < target) {
    __builtin_amdgcn_s_sleep(1);
  }
  cta.sync();
  __threadfence_system();
}

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void allReduceLsaOneShotKernel(ncclWindow_t sendWin, size_t sendOff, ncclWindow_t recvWin,
                                            size_t recvOff, size_t count, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
                                         static_cast<uint32_t>(blockIdx.x)};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const int rank = devComm.rank;
  const int nRanks = devComm.nRanks;
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t offset = static_cast<size_t>(globalTid); offset < count;
       offset += static_cast<size_t>(globalNthreads)) {
    T v = T{0};
    for (int peer = 0; peer < nRanks; peer++) {
      T* sendPtr = reinterpret_cast<T*>(ncclGetLsaPointer(sendWin, sendOff, peer));
      v = allReduceLsaSumAdd(v, sendPtr[offset]);
    }
    for (int peer = 0; peer < nRanks; peer++) {
      T* recvPtr = reinterpret_cast<T*>(ncclGetLsaPointer(recvWin, recvOff, peer));
      recvPtr[offset] = v;
    }
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void lsaAllReduceTwoShotKernel(struct ncclDevComm devComm, ncclWindow_t sendWin, size_t sendOff,
                                            ncclWindow_t recvWin, size_t recvOff, size_t countPerRank,
                                            uint64_t* reduceDoneSync, uint64_t reduceTarget, int nRanks) {
  ncclCoopCta cta;
  ncclLsaBarrierSession<ncclCoopCta> bar{cta, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
                                         static_cast<uint32_t>(blockIdx.x)};

  const size_t rankChunkStride = countPerRank * sizeof(T);
  const size_t globalElemOff = static_cast<size_t>(devComm.rank) * countPerRank;
  const size_t sliceSendByteOff = sendOff + globalElemOff * sizeof(T);
  const size_t sliceRecvByteOff = recvOff + static_cast<size_t>(devComm.rank) * rankChunkStride;
  T* reducedOut = reinterpret_cast<T*>(ncclGetLocalPointer(recvWin, sliceRecvByteOff));

  const int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int nthreads = static_cast<int>(blockDim.x * gridDim.x);
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);

  bar.sync(cta, cuda::memory_order_acquire);

  // --- Shot 1: multi-CTA LSA reduce-scatter → local recv column ---
  const size_t idxStart = static_cast<size_t>(tid) * countPerThread;
  const size_t idxEnd = countPerRank;
  const size_t idxStride = static_cast<size_t>(nthreads) * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    uint4 sum{0, 0, 0, 0};
    uint4 srcVals[2];
    *reinterpret_cast<uint4*>(&srcVals[0]) = *reinterpret_cast<const uint4*>(
      reinterpret_cast<const T*>(ncclGetLsaPointer(sendWin, sliceSendByteOff, 0)) + idx);
#pragma unroll 8
    for (int peer = 0; peer < nRanks - 1; ++peer) {
      *reinterpret_cast<uint4*>(&srcVals[(peer + 1) & 1]) = *reinterpret_cast<const uint4*>(
        reinterpret_cast<const T*>(ncclGetLsaPointer(sendWin, sliceSendByteOff, peer + 1)) + idx);
      sum = vecElementAdd<T>(sum, srcVals[peer & 1]);
    }
    sum = vecElementAdd<T>(sum, srcVals[(nRanks - 1) & 1]);
    *reinterpret_cast<uint4*>(reducedOut + idx) = sum;
  }

  allReduceTwoShotCtaArrive(reduceDoneSync, reduceTarget, cta);
  bar.sync(cta, cuda::memory_order_acquire);

  // --- Shot 2: multi-CTA LSA all-gather from local recv column ---
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    const uint4 v = *reinterpret_cast<const uint4*>(reducedOut + idx);
#pragma unroll 8
    for (int peer = 0; peer < nRanks; ++peer) {
      *reinterpret_cast<uint4*>(
        reinterpret_cast<T*>(ncclGetLsaPointer(recvWin, sliceRecvByteOff, peer)) + idx) = v;
    }
  }

  bar.sync(cta, cuda::memory_order_release);
}

__global__ void ginAllReduceResetSignalsKernel(struct ncclDevComm devComm) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.resetSignal(/*agSignal=*/0);
}

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ginAllReduceTwoShotKernel(struct ncclDevComm devComm, ncclWindow_t sendWin, size_t sendOff,
                                            ncclWindow_t recvWin, size_t recvOff, size_t countPerRank,
                                            uint64_t* reduceDoneSync, uint64_t reduceTarget, uint64_t* agDoneSync,
                                            uint64_t agTarget, int nRanks) {
  constexpr int ginContext = 0;
  constexpr unsigned kAgSignal = 0;

  ncclGin gin{devComm, ginContext};
  ncclTeam world = ncclTeamWorld(devComm);
  ncclCoopCta cta;
  ncclLsaBarrierSession<ncclCoopCta> bar{cta, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
                                         static_cast<uint32_t>(blockIdx.x)};

  const size_t rankChunkStride = countPerRank * sizeof(T);
  const size_t globalElemOff = static_cast<size_t>(devComm.rank) * countPerRank;
  const size_t sliceSendByteOff = sendOff + globalElemOff * sizeof(T);
  const size_t sliceRecvByteOff = recvOff + static_cast<size_t>(devComm.rank) * rankChunkStride;
  T* reducedOut = reinterpret_cast<T*>(ncclGetLocalPointer(recvWin, sliceRecvByteOff));
  const size_t chunkBytes = countPerRank * sizeof(T);

  const int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int nthreads = static_cast<int>(blockDim.x * gridDim.x);
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);

  bar.sync(cta, cuda::memory_order_acquire);

  // --- Shot 1: multi-CTA LSA reduce-scatter → local recv column ---
  const size_t idxStart = static_cast<size_t>(tid) * countPerThread;
  const size_t idxEnd = countPerRank;
  const size_t idxStride = static_cast<size_t>(nthreads) * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    uint4 sum{0, 0, 0, 0};
    uint4 srcVals[2];
    *reinterpret_cast<uint4*>(&srcVals[0]) = *reinterpret_cast<const uint4*>(
      reinterpret_cast<const T*>(ncclGetLsaPointer(sendWin, sliceSendByteOff, 0)) + idx);
#pragma unroll 8
    for (int peer = 0; peer < nRanks - 1; ++peer) {
      *reinterpret_cast<uint4*>(&srcVals[(peer + 1) & 1]) = *reinterpret_cast<const uint4*>(
        reinterpret_cast<const T*>(ncclGetLsaPointer(sendWin, sliceSendByteOff, peer + 1)) + idx);
      sum = vecElementAdd<T>(sum, srcVals[peer & 1]);
    }
    sum = vecElementAdd<T>(sum, srcVals[(nRanks - 1) & 1]);
    *reinterpret_cast<uint4*>(reducedOut + idx) = sum;
  }

  allReduceTwoShotCtaArrive(reduceDoneSync, reduceTarget, cta);
  bar.sync(cta, cuda::memory_order_acquire);

  // --- Shot 2: single-CTA GIN all-gather (rccl-tests HybridAlltoAllKernel CTA 0) ---
  if (blockIdx.x == 0) {
    const uint64_t signalValue = gin.readSignal(kAgSignal);
    ncclBarrierSession<ncclCoopCta> ginBar{cta, ncclTeamTagWorld(), gin, /*barrierIndex=*/0};
    ginBar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

    for (int dst = threadIdx.x; dst < nRanks; dst += blockDim.x) {
      gin.put(world, dst, recvWin, sliceRecvByteOff, recvWin, sliceRecvByteOff, chunkBytes,
              ncclGin_SignalInc{kAgSignal});
    }

    gin.waitSignal(cta, kAgSignal, signalValue + static_cast<uint64_t>(nRanks));
    gin.flush(cta);

    ginBar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
  }

  bar.sync(cta, cuda::memory_order_acquire);
  allReduceTwoShotCtaArrive(agDoneSync, agTarget, cta);
  bar.sync(cta, cuda::memory_order_release);
}

} // namespace meta::comms
