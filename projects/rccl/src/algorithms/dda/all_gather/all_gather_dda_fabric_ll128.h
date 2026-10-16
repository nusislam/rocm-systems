/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-gather device kernel for the DDA fabric path (gfx1250).
 * A warp owns one 2 KiB slice: eight uint64 registers per lane. Each block (one
 * CU) exchanges its slices with every remote peer, posting at most
 * kDdaLL128AGFanout peers at a time and draining between batches.
 * No cross-rank GPU barrier; staging uses comm->ddaScratch reached via
 * comm->ddaPeerPtrsDev.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/device/CollCommon_ll128.h"

namespace dda::common {

// Slot geometry, derived from the scratch bank the host picked. The bank splits
// evenly across ranks and the per-rank stride is floored to a whole number of
// slices, so a slot always begins on a slice boundary and the 16B wire accesses
// stay aligned however nRanks divides the bank.
constexpr size_t ddaLL128AgSlotWords(size_t bankSize, int nRanks) {
  return ddaLLSlotPkts(bankSize, sizeof(uint64_t) * (size_t)nRanks,
                       (size_t)kDdaLL128WireWordsPerSlice);
}

// Peers one CU posts before draining. A clique with more remote peers takes
// another batch; the CU never has more than this many destinations in flight.
constexpr int kDdaLL128AGFanout = 8;

// LL128 all-gather. 1D grid over slices: each block is one CU and exchanges the
// slices it owns with every remote peer. Payload is packed once per slice and
// stored to kDdaLL128AGFanout peers, then those stores drain before the next
// batch. Polls start only after every peer has been posted. Each rank's first
// batch is a different peer set, so waiting for a receive inside a send batch
// deadlocks once nPeers exceeds the fanout.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllGatherFabricLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                          T* __restrict__ recvbuff, // local user output
                                          const T* __restrict__ sendbuff, // local user input
                                          size_t perRankBytes, // per-rank payload; multiple of 16
                                          int selfRank, int nRanksRt,
                                          uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                          int epochLen, // number of cells in epochDev
                                          size_t slicesTotal, // slices this call uses
                                          size_t bankSize) { // scratch bank size (from host)

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int nPeers = nRanks - 1;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const int flatBlockId = (int)blockIdx.x;
  const int total = (int)gridDim.x;
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128AgSlotWords(bankSize, nRanks);
  // Bank 1 sits at bankSize rather than right after the slots, so every DDA LL
  // tier sharing this scratch and epoch counter banks at the same byte offsets.
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));

  // Slices stride by warp across the whole grid. Every warp covers all peers.
  const size_t gwarp = (size_t)blockIdx.x * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.x * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  const uint64_t selfSlot = (uint64_t)selfRank * (uint64_t)slotWords;

  // Phase 1: pack each slice once and push it to every remote peer,
  // kDdaLL128AGFanout at a time. Each batch drains on a store wait so the next
  // batch can be posted while other ranks are still posting.
  for (int batch = 0; batch < nPeers; batch += kDdaLL128AGFanout) {
    for (size_t s = gwarp; s < slicesTotal; s += wstride) {
      const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem = perRankBytes - dataByte;
      const int eltInSlice =
        rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
      uint64_t regs[kDdaLL128WordsPerThread];
      ddaLL128LoadRegs<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);
#pragma unroll
      for (int p = 0; p < kDdaLL128AGFanout; ++p) {
        const int peerOff = batch + p;
        if (peerOff < nPeers) {
          const int peer = (selfRank + peerOff + 1) % nRanks;
          uint64_t* scatterSlot =
            reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + selfSlot;
          ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs, flag,
                            flagLane);
        }
      }
    }
#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
    // Every wave of this CU finishes the batch before any wave posts the next,
    // so the CU has at most kDdaLL128AGFanout peers in flight.
    __syncthreads();
  }

  // Local copy sendbuff -> recvbuff[selfRank].
  {
    v4u_gptr s4 = (v4u_gptr)sendbuff;
    v4u_gptr d4 = (v4u_gptr)(reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes);
    const size_t nVec = perRankBytes >> 4; // 16B chunks
    const size_t gtid = (size_t)flatBlockId * (size_t)nthreads + (size_t)tid;
    const size_t stride = (size_t)total * (size_t)nthreads;
    for (size_t i = gtid; i < nVec; i += stride) {
      d4[i] = s4[i];
    }
  }

  // Phase 2: poll the same slices from every peer and unpack. Safe to wait now:
  // every rank has already posted every peer.
  for (int batch = 0; batch < nPeers; batch += kDdaLL128AGFanout) {
    for (size_t s = gwarp; s < slicesTotal; s += wstride) {
      const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem = perRankBytes - dataByte;
      const int eltInSlice =
        rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
      // One peer at a time: the spin waits for that peer's flag.
#pragma unroll 1
      for (int p = 0; p < kDdaLL128AGFanout; ++p) {
        const int peerOff = batch + p;
        if (peerOff < nPeers) {
          const int peer = (selfRank + peerOff + 1) % nRanks;
          const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
                                       (uint64_t)peer * (uint64_t)slotWords;
          int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;
          uint64_t vr[kDdaLL128WordsPerThread];
          ddaLL128PollWire(gatherSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, vr, flag, lane);
          ddaLL128StoreRegs<int8_t>(dstBytes + dataByte, vr, eltInSlice, lane, flagLane);
        }
      }
    }
  }
#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace dda::common
