/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL-protocol reduce-scatter device kernel for the DDA fabric path. Each 16-byte
 * line holds 8 bytes of payload and two 4-byte flags; the flags carry cross-rank
 * sync, so no GPU barrier is needed. Staging uses the DDA scratch
 * (comm->ddaScratch, reachable via comm->ddaPeerPtrsDev).
 *
 * Reduce-scatter combines the personalized scatter of all-to-all with the fold
 * of all-reduce: rank selfRank owns output shard selfRank, so recvbuff[i] =
 * sum over ranks r of sendbuff_r[selfRank*recvcount + i]. Phase 1 scatters this
 * rank's per-peer chunk (sendbuff[peer]) into peer's slot; phase 2 folds every
 * incoming chunk (plus this rank's own self-chunk) into recvbuff. Only the
 * per-peer publish source and the reduce seed differ from all_reduce_dda_ll.h.
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

namespace dda::common {

// Per-shard payload cap, enforced by eligibility. The slot stride is fixed so
// every rank addresses the same geometry. One bank is nRanks slots with the
// 8B payload expanded to 16B: 8 MiB at 1 MiB/rank and 4 ranks.
constexpr size_t kDdaLLRsMaxBytes = 1ULL*1024*1024;         // 1 MiB per-rank (= 4 MiB total at 4 ranks)
constexpr size_t kDdaLLRsSlotStridePkts = kDdaLLRsMaxBytes / 8;   // 131072

// Bytes in one bank: nRanks slots at the fixed stride.
constexpr size_t ddaLLRsBankBytes(int nRanks) {
  return (size_t)nRanks * kDdaLLRsSlotStridePkts * sizeof(LLPacket16);
}

// Packet offset of this epoch's bank. Odd epochs start at bankSize, where the
// other LL tiers sharing this scratch put bank 1.
constexpr size_t ddaLLRsBankOffsetPkts(uint32_t flag, size_t bankSize) {
  return (size_t)(flag & 1u) * (bankSize / sizeof(LLPacket16));
}

// Phase-2 reads in flight at once. Each peer keeps two data words and two
// flags, so 4 stays in registers; 8 matches the send batch but costs twice
// the temporary values.
constexpr int kDdaLLRsRecvBatch = 4;

// Peer groups inside one block, and the packets one group covers per tile.
// The launcher's block is kDdaLLRsPeerGroups * kDdaLLRsTile threads: group
// g = threadIdx.x / kDdaLLRsTile owns remote ranks r = 1+g, 1+g+G, ... and
// lane threadIdx.x % kDdaLLRsTile owns one packet of the tile. Four groups
// cut the serial poll depth by four without a second kernel launch.
constexpr int kDdaLLRsPeerGroups = 4;
constexpr int kDdaLLRsTile = 64;

// Fabric launch uses one block per CU (ddaFabricMaxBlocks is the CU count).
// LL reduce-scatter stays within 64 CUs; extra tiles stride across that grid.
constexpr int kDdaLLRsMaxBlocks = 64;

// LL reduce-scatter kernel. 1D grid over tiles of kDdaLLRsTile packets (8B
// payload each) of the per-rank shard (recvcount elements).
//
// Phase 1 (scatter): rank selfRank writes its chunk-for-peer (sendbuff[peer])
// into peer's scratch at slot selfRank, as LL lines carrying the epoch flag.
// The peer groups of a block send different peers, so those stores overlap.
// Phase 2 (reduce): each group polls the peers it owns and reduces them to a
// partial in shared memory. Group 0 seeds with this rank's own chunk
// (sendbuff[selfRank]) and folds the partials into recvbuff. Flag polling
// provides cross-rank ordering. A block barrier only folds the peer groups.
// Scratch is double-buffered: bank = flag & 1. The block is kDdaLLRsPeerGroups * kDdaLLRsTile threads.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterFabricLL(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                           T* __restrict__ recvbuff, // local user output (recvcount elems)
                                           const T* __restrict__ sendbuff, // local user input (recvcount*nRanks)
                                           size_t recvcount, // per-rank shard element count
                                           int selfRank, int nRanksRt,
                                           uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                           int epochLen, // number of cells in epochDev
                                           size_t bankSize, // scratch bank size (from host)
                                           DdaKernelParams params) {

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = recvcount * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet
  const size_t slot = kDdaLLRsSlotStridePkts;
  const size_t chunkWords = nPk * 2; // uint32 words per shard chunk
  constexpr int kGroups = kDdaLLRsPeerGroups;
  constexpr int kTile = kDdaLLRsTile;
  constexpr int kBatch = kDdaLLRsRecvBatch;
  constexpr int kStep = kGroups * kBatch;
  const int group = (int)threadIdx.x / kTile;
  const int lane = (int)threadIdx.x % kTile;

#if (defined(__gfx1250__) || defined(__gfx1250_strict__)) && __has_builtin(__builtin_amdgcn_global_prefetch)
  // Prefetch the send chunks into L2 before the epoch barrier, one 128B line per 16 lanes of group 0.
  // th 1 walks page tables. Later tiles of a striding block are not prefetched; the first pass never was.
  {
    const size_t firstPk = (size_t)blockIdx.x * (size_t)kTile + (size_t)lane;
    if (group == 0 && (lane & 15) == 0 && firstPk < nPk) {
      const char* base = reinterpret_cast<const char*>(sendbuff) + firstPk * 8;
      for (int r = 0; r < nRanks; ++r) {
        __builtin_amdgcn_global_prefetch((const __attribute__((address_space(1))) void*)(base + (size_t)r * bytes), 0x11);
      }
    }
  }
#endif
  const uint32_t flag = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const size_t bankOffsetPkts = ddaLLRsBankOffsetPkts(flag, bankSize);

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff);
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff);
  // One partial per group per lane. Group 0 folds these after every group has published.
  __shared__ uint32_t part0[kGroups * kTile];
  __shared__ uint32_t part1[kGroups * kTile];
  // volatile: every tile re-reads this after the barrier, including after another lane has aborted.
  __shared__ volatile int anyAbort;

  const size_t nTiles = (nPk + (size_t)kTile - 1) / (size_t)kTile;
  const int trips = (nRanks - 1 + kStep - 1) / kStep;

  // Phase 1, every tile, before any poll. A group that is still storing must not lose the CU to a
  // sibling group that has already started spinning on data this rank has not sent.
  for (size_t tile = (size_t)blockIdx.x; tile < nTiles; tile += (size_t)gridDim.x) {
    const size_t pk = tile * (size_t)kTile + (size_t)lane;
    const bool live = pk < nPk;
    for (int t = 0; t < trips; ++t) {
      const int r0 = 1 + group + t * kStep;
      uint32_t d0[kBatch], d1[kBatch];
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const int r = r0 + i * kGroups;
        if (live && r0 < nRanks && r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          const uint32_t* inPeer = in + (size_t)peer * chunkWords;
          d0[i] = inPeer[2 * pk];
          d1[i] = inPeer[2 * pk + 1];
        }
      }
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const int r = r0 + i * kGroups;
        if (live && r0 < nRanks && r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
          ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), d0[i], flag, d1[i], flag);
        }
      }
    }
  }

  if (threadIdx.x == 0) {
    anyAbort = 0;
  }
  __syncthreads();

  // Phase 2. Every thread reaches the barriers below: a finished group waits there while another
  // group is still polling, and an aborted lane waits there instead of returning.
  const uint32_t* inSelf = in + (size_t)selfRank * chunkWords;
  LLPacket16* myBase = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts;
  for (size_t tile = (size_t)blockIdx.x; tile < nTiles; tile += (size_t)gridDim.x) {
    const size_t pk = tile * (size_t)kTile + (size_t)lane;
    const bool live = pk < nPk;
    // The last tile's idle lanes poll its first packet, which is in range, so they stay in the
    // same waits as the lanes that own a packet.
    const size_t pkPoll = live ? pk : tile * (size_t)kTile;
    uint32_t acc0 = 0;
    uint32_t acc1 = 0;
    bool aborted = false;
    if (anyAbort == 0) {
      for (int t = 0; t < trips; ++t) {
        const int r0 = 1 + group + t * kStep;
        const uint32_t* line[kBatch];
        uint32_t d0[kBatch], d1[kBatch];
        uint32_t mask = 0;
#pragma unroll
        for (int i = 0; i < kBatch; ++i) {
          const int r = r0 + i * kGroups;
          const bool on = r0 < nRanks && r < nRanks;
          line[i] = on ? reinterpret_cast<const uint32_t*>(myBase + (size_t)((selfRank + r) % nRanks) * slot + pkPoll)
                       : nullptr;
          mask |= on ? (1u << i) : 0u;
        }
        if (mask == 0) {
          continue;
        }
        if (!ddaLLWaitLineFlagsBatched(line, mask, flag, params, d0, d1)) {
          aborted = true;
          break;
        }
        if (live) {
#pragma unroll
          for (int i = 0; i < kBatch; ++i) {
            if ((mask >> i) & 1u) {
              ddaLLAddPayload<T>(acc0, acc1, d0[i], d1[i]);
            }
          }
        }
      }
    }
    part0[group * kTile + lane] = acc0;
    part1[group * kTile + lane] = acc1;
    if (aborted) {
      atomicOr((int*)&anyAbort, 1);
    }
    __syncthreads();

    // Self, then group 0's peers, then group 1's, and so on. Groups that own no peer contribute
    // nothing; their partial stays zero and is not folded. In-place recvbuff aliases this rank's
    // own shard: phase 1 never reads that shard, and this store is the first write to it.
    if (live && group == 0 && anyAbort == 0) {
      uint32_t s0 = inSelf[2 * pk];
      uint32_t s1 = inSelf[2 * pk + 1];
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        if (1 + g < nRanks) {
          ddaLLAddPayload<T>(s0, s1, part0[g * kTile + lane], part1[g * kTile + lane]);
        }
      }
      out[2 * pk] = s0;
      out[2 * pk + 1] = s1;
    }
    __syncthreads();
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
}

} // namespace dda::common
