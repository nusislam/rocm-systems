/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Hierarchical DDA all-gather for the fabric/VMM path, after the symmetric
 * hierarchical AllGather (agHierPush in src/device/symmetric/all_gather.cuh).
 *
 * The ranks form nGroups groups of scaleIn consecutive ranks, one group per
 * physical device (the two partitions of a GPU in DPX). A rank's rail is the
 * ranks with its partition index on every device; its siblings are the other
 * ranks of its own device. Each rank pushes its own chunk directly to every
 * other rank on its rail (one hop, not a ring) and to its siblings. A chunk
 * that arrives on the rail is forwarded to the siblings, so it crosses into
 * each device once.
 *
 * As in agHierPush, the blocks are dealt out as units. A group-0 unit copies
 * the own chunk to recvbuff and the siblings with the whole block. A group-g
 * unit splits its block in halves: the pusher warps push the own chunk to the
 * rail peer g devices ahead, while the forwarder warps wait for the chunk of
 * the rail peer g devices behind and broadcast it within the device.
 *
 * User buffers are not registered, so no peer touches recvbuff: every remote
 * write lands in the receiver's scratch at the slot of the chunk's owner, and
 * the receiver copies it from there into recvbuff. The siblings' writes are
 * drained by the warps that wrote the same lines to the siblings, one chunk
 * behind their own posts.
 *
 * A receive larger than the scratch goes through it in rounds, each a window of
 * every rank's slot, alternating between the two halves of the scratch. A rank
 * overwrites a peer's half only once that peer has released the round that used
 * it two rounds before, so one round's sends overlap the previous round's drain.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

#include <utility>

namespace dda::common {

// Most ranks per physical device the kernel handles; DPX has two.
constexpr int kDdaAgHierMaxScaleIn = 4;
// 16 B loads each thread keeps in flight.
constexpr int kDdaAgHierUnroll = 8;

// How the blocks split over the nGroups devices, as ncclSymkAgHierG0Blocks / ncclSymkAgHierGroupBlocks: group 0
// copies with whole blocks where a pushing group forwards with half of each, so it gets about half as many blocks
// as each pushing group, and at least one.
__host__ __device__ constexpr int ddaAgHierG0Blocks(int nBlocks, int nGroups) {
  const int n = nBlocks / (2 * nGroups - 1);
  return n > 1 ? n : 1;
}
__host__ __device__ constexpr int ddaAgHierGroupBlocks(int nBlocks, int nGroups) {
  const int n = (nBlocks - ddaAgHierG0Blocks(nBlocks, nGroups)) / (nGroups - 1);
  return n > 1 ? n : 1;
}

// Loads 16 B lines lo, lo + stride, ... < hi of src, then hands each line to store.
// Groups of kDdaAgHierUnroll lines are loaded before any of them is stored. SysLoad: src is scratch written by
// another GPU, read with system-scope loads.
template <bool SysLoad, typename Store>
__device__ __forceinline__ void ddaAgHierScatter(const uint4* src, size_t lo, size_t hi, size_t stride, Store store) {
  constexpr int kUnroll = kDdaAgHierUnroll;
  auto load = [&](size_t line, uint4& v) {
    if constexpr (SysLoad) {
      ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(src + line), v.x, v.y, v.z, v.w);
    } else {
      v = src[line];
    }
  };
  size_t i = lo;
  for (; i + (kUnroll - 1) * stride < hi; i += kUnroll * stride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) load(i + u * stride, v[u]);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) store(i + u * stride, v[u]);
  }
  for (; i < hi; i += stride) {
    uint4 v;
    load(i, v);
    store(i, v);
  }
}

// Software barrier over the waves of one half of a block, on a counter in LDS that starts at 0. Each wave adds one per
// call and waits for gen * nWaves, so a wave cannot run a call ahead and be counted in the previous one.
struct DdaAgHierHalf {
  uint32_t* counter;
  uint32_t nWaves;
  uint32_t gen;

  __device__ __forceinline__ void sync() {
    ++gen;
    if (threadIdx.x % warpSize == 0) {
      __scoped_atomic_fetch_add(counter, 1u, __ATOMIC_RELEASE, __MEMORY_SCOPE_WRKGRP);
      while (__scoped_atomic_load_n(counter, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_WRKGRP) < gen * nWaves) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __builtin_amdgcn_wave_barrier();
  }
};

// count is the per-rank chunk in elements, a multiple of 16 B. A rank's chunk goes in rounds of windowLines 16 B
// lines (the last may be shorter). Round r uses scratch bank r % 2, nRanks slots of windowLines lines each written by
// one rank: a rank pushes its own slot to every rail peer and every sibling, and forwards each rail slot it receives
// to its siblings. The rank's own slot is never written. With a single round only bank 0 is used, so the host may pass
// windowLines = count lines whenever nRanks slots of it fit the scratch, and half the scratch per slot set otherwise.
//
// Units and chunks depend only on windowLines, the grid and maxChunkLines, so block b of every rank runs the same
// units in the same order in every round, and the flags pair blocks of equal index. A unit's window is cut into
// chunks of at most maxChunkLines lines; the block's chunk i is chunk i * nUnitBlocks + groupBlock, clipped to the
// round. Block b of a rank therefore reads only the scratch lines block b of its rail peers and siblings write, so
// bank reuse is per block: after round r a block posts release r to the ranks that write to it, and before round r it
// waits for release r - 2 from the ranks it writes to (the same ranks). Flag values run on across the block's units
// and rounds (seq), so the posts one block makes to one rank only grow:
//   group 0:  the block copies its chunk i to recvbuff and the siblings and posts seq + i to each sibling;
//   group g:  the pushers write chunk i to rail peer g ahead and post seq + i to it; the forwarders wait for seq + i
//             from rail peer g behind, copy the chunk to recvbuff and the siblings, and post seq + i to each sibling.
// Each also drains the siblings' chunk i - 1 of the same unit. A flag slot has one writer: pushers post only to rail
// peers, the rest only to siblings. Pushers never wait, forwarders wait on pushers, and sibling drains wait on
// group-0 copies or forwarders of the same unit, and a release wait of round r only on rounds up to r - 2, which
// never wait on round r, so nothing can deadlock.
template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllGatherFabricHier(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff,
                                         const T* __restrict__ sendbuff, size_t count, int selfRank, int nRanks,
                                         int scaleIn, size_t maxChunkLines, size_t windowLines,
                                         FabricGpuBarrier barrier) {
  using FlagType = FabricGpuBarrier::FlagType;
  __shared__ uint32_t halfArrivals[2];

  const size_t nLines = count * sizeof(T) / sizeof(uint4);
  const size_t nRounds = (nLines + windowLines - 1) / windowLines;
  const int nBlocks = static_cast<int>(gridDim.x);
  const int block = static_cast<int>(blockIdx.x);
  const int nGroups = nRanks / scaleIn;
  const int card = selfRank / scaleIn;
  const int local = selfRank - card * scaleIn;
  // Card `delta` steps from this one, wrapped.
  auto cardAt = [&](int delta) {
    int c = card + delta;
    if (c >= nGroups) c -= nGroups;
    else if (c < 0) c += nGroups;
    return c;
  };
  // Sibling d (1 <= d < scaleIn) is the rank d partitions after this one on its device.
  auto sibling = [&](int d) {
    int l = local + d;
    if (l >= scaleIn) l -= scaleIn;
    return card * scaleIn + l;
  };

  const int nG0Blocks = ddaAgHierG0Blocks(nBlocks, nGroups);
  const int nGroupBlocks = ddaAgHierGroupBlocks(nBlocks, nGroups);
  // Units below nUnitRows hold a block of every group, the rest a block of each pushing group.
  const int nUnitRows = nGroups * nG0Blocks;
  const int nUnits = nG0Blocks + (nGroups - 1) * nGroupBlocks;

  // Half of each block pushes and half forwards in a group-g unit, threads numbered over the half.
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int nHalfWarps = static_cast<int>(blockDim.x) / warpSize / 2;
  const bool pusher = warp < nHalfWarps;
  const bool forwarder = !pusher && warp < 2 * nHalfWarps;
  const int ht = (pusher ? warp : warp - nHalfWarps) * warpSize + lane;
  const int htn = nHalfWarps * warpSize;

  // Chunk size and count of a window for units of nUnitBlocks blocks, chunks a multiple of align lines so the
  // threads split them evenly.
  auto chunking = [&](int nUnitBlocks, size_t align) -> std::pair<size_t, size_t> {
    const size_t perBlock = (windowLines + nUnitBlocks - 1) / nUnitBlocks;
    size_t nBlockChunksMax = (perBlock + maxChunkLines - 1) / maxChunkLines;
    if (nBlockChunksMax == 0) nBlockChunksMax = 1;
    const size_t want = nBlockChunksMax * nUnitBlocks;
    const size_t chunkLines = ((windowLines + want - 1) / want + align - 1) / align * align;
    return {chunkLines, (windowLines + chunkLines - 1) / chunkLines};
  };
  const std::pair<size_t, size_t> g0Chunking = chunking(nG0Blocks, blockDim.x);
  const std::pair<size_t, size_t> gChunking = chunking(nGroupBlocks, static_cast<size_t>(htn));
  const size_t g0ChunkLines = g0Chunking.first;
  const size_t g0Chunks = g0Chunking.second;
  const size_t gChunkLines = gChunking.first;
  const size_t gChunks = gChunking.second;
  auto blockChunks = [](int groupBlock, int nUnitBlocks, size_t nChunks) -> unsigned {
    return static_cast<size_t>(groupBlock) < nChunks ?
             static_cast<unsigned>((nChunks - groupBlock + nUnitBlocks - 1) / nUnitBlocks) :
             0u;
  };
  // Upper bound on any block's posts, the same for every block, so epochEnd advances every epoch cell alike.
  const unsigned g0Max = static_cast<unsigned>((g0Chunks + nG0Blocks - 1) / nG0Blocks);
  const unsigned gMax = static_cast<unsigned>((gChunks + nGroupBlocks - 1) / nGroupBlocks);
  const unsigned seqBound = static_cast<unsigned>(nRounds) *
                            static_cast<unsigned>((nUnits + nBlocks - 1) / nBlocks) * (g0Max > gMax ? g0Max : gMax);

  uint4* const recvLines = reinterpret_cast<uint4*>(recvbuff);
  const uint4* const selfScratch = reinterpret_cast<const uint4*>(ipcbuffs[selfRank]);
  auto scratchOf = [&](int rank) { return reinterpret_cast<uint4*>(ipcbuffs[rank]); };

  if (threadIdx.x == 0) {
    halfArrivals[0] = 0;
    halfArrivals[1] = 0;
  }
  const FlagType epoch = barrier.epochBegin();
  // The rail peers and the siblings: the ranks this rank writes to, and the only ranks that write to it.
  const int nPeers = nGroups - 1 + scaleIn - 1;
  auto peerOf = [&](int i) {
    return i < nGroups - 1 ? cardAt(i + 1) * scaleIn + local : sibling(i - (nGroups - 1) + 1);
  };
  // This rank's reads of its scratch in the previous collective must be done before a peer overwrites it. Only the
  // rail peers and the siblings write here, and this rank writes only to theirs, so the handshake is with them
  // alone rather than with every rank. Other ranks' reads of this scratch in a previous collective are done once
  // that kernel has finished here: every DDA kernel that reads a peer's scratch ends with an all-rank exit barrier.
  // The closing __syncthreads also publishes the counters above.
  barrier.epochSignalWaitPeers(nPeers, peerOf, epoch);
  DdaAgHierHalf half{&halfArrivals[pusher ? 0 : 1], static_cast<uint32_t>(nHalfWarps), 0};
  unsigned seq = 0;

  for (size_t round = 0; round < nRounds; ++round) {
    // The bank this round writes was read in round - 2; every rank written here must have released it.
    if (round >= 2) barrier.epochWaitReleased(nPeers, peerOf, epoch, static_cast<unsigned>(round - 2));

    const size_t roundLo = round * windowLines;
    const size_t roundLines = nLines - roundLo < windowLines ? nLines - roundLo : windowLines;
    const size_t bank = (round & 1) * static_cast<size_t>(nRanks) * windowLines;
    const uint4* const sendLines = reinterpret_cast<const uint4*>(sendbuff) + roundLo;
    // Where the round's lines of rank `owner` go in scratch and in recvbuff.
    auto scratchSlot = [&](int owner) { return bank + static_cast<size_t>(owner) * windowLines; };
    auto recvSlot = [&](int owner) { return static_cast<size_t>(owner) * nLines + roundLo; };
    const size_t ownScratch = scratchSlot(selfRank);
    const size_t ownRecv = recvSlot(selfRank);

    // Stores one line into recvbuff at recvOff and into every sibling's scratch at scratchOff.
    auto storeRecvAndSiblings = [&](size_t recvOff, size_t scratchOff, size_t line, uint4 v) {
      recvLines[recvOff + line] = v;
#pragma unroll
      for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
        if (d < scaleIn) scratchOf(sibling(d))[scratchOff + line] = v;
      }
    };
    // Copies lines [lo, hi) of rank ownerOf(sibLocal)'s slot, for every sibling, from this rank's scratch into
    // recvbuff, once that sibling has posted `piece`. Wave: the calling threads are one half, which waits per wave.
    auto drainSiblings = [&](unsigned piece, size_t lo, size_t hi, int t, int tn, bool wave, auto&& ownerOf) {
#pragma unroll
      for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
        if (d < scaleIn) {
          const int sib = sibling(d);
          if (wave) {
            barrier.epochWaitPieceWave(barrier.epochFlagFrom(sib), epoch, piece);
          } else {
            barrier.epochWaitPiece(barrier.epochFlagFrom(sib), epoch, piece);
          }
          const int owner = ownerOf(sib - card * scaleIn);
          uint4* const dst = recvLines + recvSlot(owner);
          ddaAgHierScatter<true>(selfScratch + scratchSlot(owner), lo + t, hi, tn,
                                 [&](size_t line, uint4 v) { dst[line] = v; });
        }
      }
    };

    for (int unit = block; unit < nUnits; unit += nBlocks) {
      const int group = unit < nUnitRows ? unit % nGroups : 1 + (unit - nUnitRows) % (nGroups - 1);
      const int groupBlock = unit < nUnitRows ? unit / nGroups : nG0Blocks + (unit - nUnitRows) / (nGroups - 1);

      if (group == 0) {
        const unsigned n = blockChunks(groupBlock, nG0Blocks, g0Chunks);
        const int t = static_cast<int>(threadIdx.x);
        const int tn = static_cast<int>(blockDim.x);
        auto range = [&](unsigned i) -> std::pair<size_t, size_t> {
          const size_t lo = (static_cast<size_t>(i) * nG0Blocks + groupBlock) * g0ChunkLines;
          return {lo, lo + g0ChunkLines < roundLines ? lo + g0ChunkLines : roundLines};
        };
        for (unsigned i = 0; i <= n; ++i) {
          if (i < n) {
            auto [lo, hi] = range(i);
            ddaAgHierScatter<false>(sendLines, lo + t, hi, tn,
                                    [&](size_t line, uint4 v) { storeRecvAndSiblings(ownRecv, ownScratch, line, v); });
            barrier.orderDestWrites();
#pragma unroll
            for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
              if (d < scaleIn) barrier.epochPostPieceDrained(sibling(d), epoch, seq + i);
            }
          }
          if (i > 0) {
            auto [lo, hi] = range(i - 1);
            drainSiblings(seq + i - 1, lo, hi, t, tn, /*wave=*/false,
                          [&](int sibLocal) { return card * scaleIn + sibLocal; });
          }
        }
        seq += n;
        __syncthreads();
        continue;
      }

      const unsigned n = blockChunks(groupBlock, nGroupBlocks, gChunks);
      auto range = [&](unsigned i) -> std::pair<size_t, size_t> {
        const size_t lo = (static_cast<size_t>(i) * nGroupBlocks + groupBlock) * gChunkLines;
        return {lo, lo + gChunkLines < roundLines ? lo + gChunkLines : roundLines};
      };
      if (pusher) {
        const int to = cardAt(group) * scaleIn + local;
        uint4* const toSlot = scratchOf(to) + ownScratch;
        for (unsigned i = 0; i < n; ++i) {
          auto [lo, hi] = range(i);
          ddaAgHierScatter<false>(sendLines, lo + ht, hi, htn, [&](size_t line, uint4 v) { toSlot[line] = v; });
          FabricGpuBarrier::drainWaveWrites();
          half.sync();
          if (ht == 0) barrier.epochStorePiece(to, epoch, seq + i);
        }
      } else if (forwarder) {
        const int fromCard = cardAt(-group);
        const int from = fromCard * scaleIn + local;
        const size_t fromScratch = scratchSlot(from);
        const size_t fromRecv = recvSlot(from);
        for (unsigned i = 0; i <= n; ++i) {
          if (i < n) {
            auto [lo, hi] = range(i);
            barrier.epochWaitPieceWave(barrier.epochFlagFrom(from), epoch, seq + i);
            ddaAgHierScatter<true>(selfScratch + fromScratch, lo + ht, hi, htn, [&](size_t line, uint4 v) {
              storeRecvAndSiblings(fromRecv, fromScratch, line, v);
            });
            FabricGpuBarrier::drainWaveWrites();
            half.sync();
            if (ht == 0) {
#pragma unroll
              for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
                if (d < scaleIn) barrier.epochStorePiece(sibling(d), epoch, seq + i);
              }
            }
          }
          // The siblings' forwarders of this unit forwarded the chunk of their own rail peer on fromCard.
          if (i > 0) {
            auto [lo, hi] = range(i - 1);
            drainSiblings(seq + i - 1, lo, hi, ht, htn, /*wave=*/true,
                          [&](int sibLocal) { return fromCard * scaleIn + sibLocal; });
          }
        }
      }
      seq += n;
      __syncthreads();
    }

    // Done reading this round's bank: the ranks that write here may reuse it two rounds on.
    if (round + 2 < nRounds) barrier.epochPostReleased(nPeers, peerOf, epoch, static_cast<unsigned>(round));
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
  barrier.epochEnd(epoch, seqBound);
}

} // namespace dda::common
