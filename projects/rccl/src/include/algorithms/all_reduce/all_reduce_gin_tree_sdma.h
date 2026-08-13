/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Device kernel: GIN put-based tree all-reduce (reduce-scatter + all-gather).
 *
 * CTA 0 performs full-chunk RS and AG data transfers (one GIN signal and one
 * rail barrier index each). All CTAs participate in the local reduction;
 * scratch-resident atomics rendezvous CTAs around the transfer phases.
 ************************************************************************/

#pragma once

#include "nccl_device.h"
#include "nccl_device/ptr.h"
#include "algorithms/CollCommon.h"

namespace meta::comms {

__device__ __forceinline__ void ginAllReduceCtaArrive(uint64_t* counter, ncclCoopCta cta) {
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(counter, 1ULL, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  cta.sync();
  if (threadIdx.x == 0) {
    const uint64_t target = static_cast<uint64_t>(gridDim.x);
    while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < target) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  cta.sync();
  __threadfence_system();
}

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ginAllReduceTreeKernel(struct ncclDevComm devComm, ncclWindow_t sendWin, size_t sendOff,
                                         ncclWindow_t scratchWin, size_t scratchOff, ncclWindow_t recvWin,
                                         size_t recvOff, size_t count, int nRanks, int rank, unsigned rsSignalBase,
                                         unsigned agSignalBase) {
  constexpr int ginContext = 0;
  constexpr unsigned kTransferCta = 0;
  ncclGin gin{devComm, ginContext};
  ncclTeam world = ncclTeamWorld(devComm);
  ncclCoopCta cta;

  const size_t countPerRank = count / (size_t)nRanks;
  const size_t chunkBytes = countPerRank * sizeof(T);

  const unsigned rsSig = rsSignalBase;
  const unsigned agSig = agSignalBase;

  ncclSymPtr<T> scratch{scratchWin, scratchOff};
  T* scratchBase = scratch.localPtr();
  const size_t scratchDataBytes = (size_t)nRanks * chunkBytes;
  uint64_t* rsCtaSync = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(scratchBase) + scratchDataBytes);
  uint64_t* agCtaSync = rsCtaSync + 1;
  uint64_t* agDoneSync = agCtaSync + 1;

  const int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int nthreads = static_cast<int>(blockDim.x * gridDim.x);

  // --- Phase 1: single-CTA reduce-scatter ---
  if (blockIdx.x == kTransferCta) {
    ncclGinBarrierSession<ncclCoopCta> bar{cta, gin, world, devComm.railGinBarrier, kTransferCta};
    const uint64_t rsSigVal = gin.readSignal(rsSig);

    bar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

    for (int dst = tid; dst < nRanks; dst += static_cast<int>(blockDim.x)) {
      gin.put(world, dst, scratchWin, scratchOff + (size_t)rank * chunkBytes, sendWin,
              sendOff + (size_t)dst * chunkBytes, chunkBytes, ncclGin_SignalInc{rsSig}, ncclGin_None{},
              ncclCoopThread{});
    }
    gin.waitSignal(cta, rsSig, rsSigVal + (uint64_t)nRanks);
    gin.flush(cta);
    bar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
  }
  ginAllReduceCtaArrive(rsCtaSync, cta);

  // --- Phase 2: multi-CTA local reduction ---
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const size_t idxStart = static_cast<size_t>(tid) * countPerThread;
  const size_t idxEnd = countPerRank;
  const size_t idxStride = static_cast<size_t>(nthreads) * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    uint4 sum{0, 0, 0, 0};
    uint4 srcVals[2];
    *reinterpret_cast<uint4*>(&srcVals[0]) =
      *reinterpret_cast<const uint4*>(scratchBase + 0 * countPerRank + idx);
#pragma unroll 8
    for (int r = 0; r < nRanks - 1; ++r) {
      *reinterpret_cast<uint4*>(&srcVals[(r + 1) & 1]) =
        *reinterpret_cast<const uint4*>(scratchBase + (size_t)(r + 1) * countPerRank + idx);
      sum = vecElementAdd<T>(sum, srcVals[r & 1]);
    }
    sum = vecElementAdd<T>(sum, srcVals[(nRanks - 1) & 1]);
    *reinterpret_cast<uint4*>(scratchBase + (size_t)rank * countPerRank + idx) = sum;
  }

  ginAllReduceCtaArrive(agCtaSync, cta);

  // --- Phase 3: single-CTA all-gather ---
  if (blockIdx.x == kTransferCta) {
    ncclGinBarrierSession<ncclCoopCta> bar{cta, gin, world, devComm.railGinBarrier, kTransferCta};
    const uint64_t agSigVal = gin.readSignal(agSig);

    for (int dst = tid; dst < nRanks; dst += static_cast<int>(blockDim.x)) {
      gin.put(world, dst, recvWin, recvOff + (size_t)rank * chunkBytes, scratchWin,
              scratchOff + (size_t)rank * chunkBytes, chunkBytes, ncclGin_SignalInc{agSig}, ncclGin_None{},
              ncclCoopThread{});
    }
    gin.waitSignal(cta, agSig, agSigVal + (uint64_t)nRanks);
    gin.flush(cta);
    bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
  }
  ginAllReduceCtaArrive(agDoneSync, cta);
}

} // namespace meta::comms
