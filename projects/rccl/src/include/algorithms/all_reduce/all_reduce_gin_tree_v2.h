/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Device kernel: GIN put-based tree all-reduce (reduce-scatter + all-gather).
 ************************************************************************/

#pragma once

#include "nccl_device.h"
#include "nccl_device/ptr.h"
#include "algorithms/CollCommon.h"

namespace meta::comms {

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ginAllReduceTreeKernel(struct ncclDevComm devComm, ncclWindow_t sendWin, size_t sendOff,
                                         ncclWindow_t scratchWin, size_t scratchOff, ncclWindow_t recvWin,
                                         size_t recvOff, size_t count, int nRanks, int rank, unsigned rsSignalBase,
                                         unsigned agSignalBase) {
  constexpr int ginContext = 0;
  ncclGin gin{devComm, ginContext};
  ncclTeam world = ncclTeamWorld(devComm);
  ncclCoopCta cta;

  const size_t countPerRank = count / (size_t)nRanks;
  const size_t chunkBytes = countPerRank * sizeof(T);

  const unsigned rsSig = rsSignalBase + blockIdx.x;
  const unsigned agSig = agSignalBase + blockIdx.x;
  const uint64_t rsSigVal = gin.readSignal(rsSig);
  const uint64_t agSigVal = gin.readSignal(agSig);

  ncclGinBarrierSession<ncclCoopCta> bar{cta, gin, world, devComm.railGinBarrier, blockIdx.x};
  bar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  const int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int nthreads = static_cast<int>(blockDim.x * gridDim.x);
  const int receivingCta = (rank % nthreads) / static_cast<int>(blockDim.x);

  // Phase 1: reduce-scatter via remote writes into the DDA scratch buffer.
  // Rank `rank` writes chunk `dst` from sendbuff into slot `rank` on peer `dst`.
  for (int dst = tid; dst < nRanks; dst += nthreads) {
    gin.put(world, dst, scratchWin, scratchOff + (size_t)rank * chunkBytes, sendWin,
            sendOff + (size_t)dst * chunkBytes, chunkBytes, ncclGin_SignalInc{rsSig}, ncclGin_None{},
            ncclCoopThread{});
  }

  if (blockIdx.x == receivingCta) {
    gin.waitSignal(cta, rsSig, rsSigVal + (uint64_t)nRanks);
  }
  bar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  ncclSymPtr<T> scratch{scratchWin, scratchOff};
  T* scratchBase = scratch.localPtr();

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

  bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);

  // Phase 2: all-gather via remote writes into the symmetric receive buffer.
  for (int dst = tid; dst < nRanks; dst += nthreads) {
    gin.put(world, dst, recvWin, recvOff + (size_t)rank * chunkBytes, scratchWin,
            scratchOff + (size_t)rank * chunkBytes, chunkBytes, ncclGin_SignalInc{agSig}, ncclGin_None{},
            ncclCoopThread{});
  }

  if (blockIdx.x == receivingCta) {
    gin.waitSignal(cta, agSig, agSigVal + (uint64_t)nRanks);
  }

  gin.flush(cta);
  bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

} // namespace meta::comms
