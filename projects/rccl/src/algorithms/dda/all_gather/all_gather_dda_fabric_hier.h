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
 * that arrives on the rail is read by the siblings straight out of the
 * receiver's scratch, so it crosses into each device once.
 *
 * As in agHierPush, the blocks are dealt out as units. A group-0 unit copies
 * the own chunk to recvbuff and the siblings with the whole block. A group-g
 * unit splits its block in halves: the pusher warps push the own chunk to the
 * rail peer g devices ahead, while the forwarder warps wait for the chunk of
 * the rail peer g devices behind, tell the siblings it is there and copy it to
 * recvbuff.
 *
 * User buffers are not registered, so no peer touches recvbuff: every remote
 * write lands in the receiver's scratch at the slot of the chunk's owner, and
 * the receiver copies it from there into recvbuff. Own chunks pushed by a
 * sibling are drained from this rank's scratch, rail chunks received by a
 * sibling from that sibling's scratch. That drain runs while the next chunk is
 * in flight: a group-0 block drains between issuing a chunk's stores and
 * waiting for them, and the forwarders drain before waiting for the next rail
 * chunk.
 *
 * A receive larger than the scratch goes through it in rounds, each a window of
 * every rank's slot, alternating between the two halves of the scratch. A rank
 * overwrites a peer's half only once every rank that reads it has released the
 * round that used it two rounds before, so one round's sends overlap the
 * previous round's drain.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "nccl_device/coop.h"

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

// Spins until `word` reaches `value`. One thread calls it.
__device__ __forceinline__ void agHierWait(uint32_t* abortFlag, FabricGpuBarrier::FlagType* word,
                                           FabricGpuBarrier::FlagType value) {
  cuda::atomic_ref<FabricGpuBarrier::FlagType> ref(*word);
  uint32_t steps = 0;
  while (ref.load(cuda::memory_order_relaxed) < value) {
    if (nccl::utility::testAbort(abortFlag, steps)) break;
  }
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
}

// Spins until this block's flag reaches `value` on every other rank of the package. One thread calls it.
__device__ __forceinline__ void agHierWaitPackage(uint32_t* abortFlag, const FabricGpuBarrier& barrier, int scaleIn,
                                                 int local, int card, FabricGpuBarrier::FlagType value) {
  for (int r = 0; r < scaleIn; r++) {
    if (r == local) continue;
    agHierWait(abortFlag, const_cast<FabricGpuBarrier::FlagType*>(barrier.epochFlagFrom(card * scaleIn + r)), value);
  }
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

// count is the per-rank chunk in elements, a multiple of 16 B. A rank's chunk goes in rounds of windowLines 16 B
// lines (the last may be shorter). Round r uses scratch bank r % 2, nRanks slots of windowLines lines each written by
// one rank: a rank pushes its own slot to every rail peer and every sibling. A rail slot it receives is read by its
// siblings in place. The rank's own slot is never written. With a single round only bank 0 is used, so the host may
// pass windowLines = count lines whenever nRanks slots of it fit the scratch, and half the scratch per slot set
// otherwise.
//
// Units and chunks depend only on windowLines, the grid and maxChunkLines, so block b of every rank runs the same
// units in the same order in every round, and the flags pair blocks of equal index. A unit's window is cut into
// chunks of at most maxChunkLines lines; the block's chunk i is chunk i * nUnitBlocks + groupBlock, clipped to the
// round. Block b of a rank therefore reads only scratch lines that block b of other ranks write, so bank reuse is per
// block. A rank's scratch is read by the rank and by its siblings, so the ranks reading what a rank writes (its rail
// peers and their siblings, and its own siblings) are every other rank: after round r a block posts release r to
// every other rank, and before round r it waits for release r - 2 from every other rank. Flag values run on across
// the block's units and rounds (seq), so the posts one block makes to one rank only grow:
//   group 0:  the block copies its chunk i to recvbuff and the siblings and posts seq + i to each sibling;
//   group g:  the pushers write chunk i to rail peer g ahead and post seq + i to it; the forwarders wait for seq + i
//             from rail peer g behind, post seq + i to each sibling, and copy the chunk to recvbuff.
// The group-0 block drains the siblings' pushed chunk i - 1 from this rank's scratch after issuing chunk i's stores
// and before posting chunk i; the forwarders pull the siblings' rail chunk i - 1, from the siblings' scratch, before
// waiting for rail chunk i. A flag slot has one writer: pushers post only to rail peers, the rest only to siblings.
// A post of chunk i waits only on a sibling post of chunk i - 1 (group 0) or on the rail pusher's post of chunk i
// (forwarders), a forwarder's pull of chunk i - 1 waits on the sibling's post of that same chunk, and a release wait
// of round r only on rounds up to r - 2. Each of those is an earlier post, so the waits cannot form a cycle.
template <typename T>
__launch_bounds__(512)
  __global__ void ddaAllGatherFabricHier(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff,
                                         const T* __restrict__ sendbuff, size_t count, int selfRank, int nRanks,
                                         int scaleIn, size_t maxChunkLines, size_t windowLines,
                                         FabricGpuBarrier barrier) {
  using FlagType = FabricGpuBarrier::FlagType;

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
  // WARP_SIZE, not warpSize: ncclCoopWarpSpan counts warps the same way (32 on gfx1250).
  const int lane = static_cast<int>(threadIdx.x) % WARP_SIZE;
  const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int nHalfWarps = static_cast<int>(blockDim.x) / WARP_SIZE / 2;
  const bool pusher = warp < nHalfWarps;
  const bool forwarder = !pusher && warp < 2 * nHalfWarps;
  const int ht = (pusher ? warp : warp - nHalfWarps) * WARP_SIZE + lane;
  const int htn = nHalfWarps * WARP_SIZE;
  // Same half-block barrier as agHierPush. Ids 0 and 1 are this block's two halves.
  ncclCoopWarpSpan half{/*warp0=*/pusher ? 0 : nHalfWarps, /*nWarps=*/nHalfWarps, /*id=*/pusher ? 0 : 1};

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

  // Zeros the LDS slots ncclCoopWarpSpan::sync uses. Every thread must take part.
  ncclCoopNamedBarrierInit();
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
  barrier.epochSignalWaitPeers(nPeers, peerOf, epoch);
  // Every other rank: the readers of what this rank writes within a launch, and the writers of what it reads.
  auto otherRank = [&](int i) { return i < selfRank ? i : i + 1; };
  unsigned seq = 0;

  for (size_t round = 0; round < nRounds; ++round) {
    // The bank this round writes was read in round - 2; every rank reading it must have released it.
    if (round >= 2) barrier.epochWaitReleased(nRanks - 1, otherRank, epoch, static_cast<unsigned>(round - 2));

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
        // Chunk i's stores go out first. The siblings' chunk i - 1 is copied out of scratch while they are in flight,
        // and only then does the block sync and post chunk i.
        for (unsigned i = 0; i <= n; ++i) {
          if (i < n) {
            auto [lo, hi] = range(i);
            ddaAgHierScatter<false>(sendLines, lo + t, hi, tn,
                                    [&](size_t line, uint4 v) { storeRecvAndSiblings(ownRecv, ownScratch, line, v); });
          }
          if (i > 0) {
            auto [lo, hi] = range(i - 1);
            if (t == 0) agHierWaitPackage(barrier.abortFlag(), barrier, scaleIn, local, card, epoch + 2 + (seq + i - 1));
            __syncthreads();
#pragma unroll
            for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
              if (d < scaleIn) {
                const int sib = sibling(d);
                const uint4* const src = selfScratch + scratchSlot(sib);
                uint4* const dst = recvLines + recvSlot(sib);
                ddaAgHierScatter<true>(src, lo + t, hi, tn, [&](size_t line, uint4 v) { dst[line] = v; });
              }
            }
          }
          if (i < n) {
            // Orders every warp's stores before thread 0's system-scope release fence and flag store.
            __syncthreads();
#pragma unroll
            for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
              if (d < scaleIn) barrier.epochPostPieceDrained(sibling(d), epoch, seq + i);
            }
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
          // Orders every pusher warp's stores before the leader's system-scope flag post.
          half.sync();
          if (ht == 0) barrier.epochStorePiece(to, epoch, seq + i);
        }
      } else if (forwarder) {
        const int fromCard = cardAt(-group);
        const int from = fromCard * scaleIn + local;
        uint4* const fromDst = recvLines + recvSlot(from);
        const uint4* const fromSrc = selfScratch + scratchSlot(from);
        // The siblings' chunk i - 1 is pulled while the rail peer is still storing chunk i, and the wait for chunk i
        // starts when that pull is done.
        for (unsigned i = 0; i <= n; ++i) {
          if (i > 0) {
            auto [lo, hi] = range(i - 1);
            if (ht == 0) {
              agHierWaitPackage(barrier.abortFlag(), barrier, scaleIn, local, card, epoch + 2 + (seq + i - 1));
            }
            half.sync();
#pragma unroll
            for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
              if (d < scaleIn) {
                const int sib = sibling(d);
                const int owner = fromCard * scaleIn + (sib - card * scaleIn);
                const uint4* const src = scratchOf(sib) + scratchSlot(owner);
                uint4* const dst = recvLines + recvSlot(owner);
                ddaAgHierScatter<true>(src, lo + ht, hi, htn, [&](size_t line, uint4 v) { dst[line] = v; });
              }
            }
          }
          if (i < n) {
            auto [lo, hi] = range(i);
            if (ht == 0) {
              agHierWait(barrier.abortFlag(), const_cast<FlagType*>(barrier.epochFlagFrom(from)), epoch + 2 + (seq + i));
            }
            half.sync();
            // The chunk is in this rank's scratch, where the siblings read it.
            if (ht == 0) {
#pragma unroll
              for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
                if (d < scaleIn) barrier.epochStorePiece(sibling(d), epoch, seq + i);
              }
            }
            ddaAgHierScatter<true>(fromSrc, lo + ht, hi, htn, [&](size_t line, uint4 v) { fromDst[line] = v; });
          }
        }
      }
      seq += n;
      __syncthreads();
    }

    // Done reading this round's bank, here and in the siblings: the writers may reuse it two rounds on.
    if (round + 2 < nRounds) {
      barrier.epochPostReleased(nRanks - 1, otherRank, epoch, static_cast<unsigned>(round));
    }
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
  barrier.epochEnd(epoch, seqBound);
}

} // namespace dda::common
