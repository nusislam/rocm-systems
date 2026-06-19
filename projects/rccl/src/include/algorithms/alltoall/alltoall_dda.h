/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
 * Adapted for alltoall collective operation.
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
    __global__ void ddaAllToAllIpc(
        T* const* __restrict__ ipcbuffs,
        T* __restrict__ recvbuff,
        size_t count,
        const T* __restrict__ sendbuff,
        int selfRank,
        IpcGpuBarrier barrier) {
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t copyCount = count * NRANKS;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // It is expensive to launch hipMemcpyAsync on ROCm
  // Move data copy here. Each block copies part of sendbuff data
  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, copyCount, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      int srcRank = r;
      int srcIdx = idx + selfRank * idxEnd;
      int destIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[destIdx]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();
}

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllToAllPushIpc(
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

  // Step 1: Push data directly to remote IPC buffers
  // Each rank writes its data for destRank into destRank's IPC buffer
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int destRank = 0; destRank < NRANKS; ++destRank) {
      size_t srcOffset = destRank * count + idx;  // Data for destRank in send buffer
      size_t destOffset = selfRank * count + idx; // Where selfRank's data goes in destRank's buffer

      if (srcOffset < count * NRANKS && destOffset < count * NRANKS) {
        // Write directly to destRank's IPC buffer at the position for selfRank's data
        *reinterpret_cast<uint4*>(&ipcbuffs[destRank][destOffset]) =
            reinterpret_cast<const uint4*>(&sendbuff[srcOffset])[0];
      }
    }
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */,
      true /* prevRemoteWrite */>();

  // Step 2: Local copy from own IPC buffer to recv buffer
  // After all ranks have written to this rank's IPC buffer, copy locally
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int srcRank = 0; srcRank < NRANKS; ++srcRank) {
      size_t ipcOffset = srcRank * count + idx;  // Data from srcRank in local IPC buffer
      size_t destOffset = srcRank * count + idx; // Where to place data from srcRank

      if (ipcOffset < count * NRANKS && destOffset < count * NRANKS) {
        *reinterpret_cast<uint4*>(&recvbuff[destOffset]) =
            reinterpret_cast<const uint4*>(&ipcbuffs[selfRank][ipcOffset])[0];
      }
    }
  }

  barrier.syncOnSameBlockIdx<
      true  /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */,
      false /* prevRemoteWrite */>();
}


} // namespace meta::comms

