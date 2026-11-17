/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Hierarchical DDA all-gather for the fabric/VMM path, after the symmetric
 * hierarchical AllGather (agHierPush in src/device/symmetric/all_gather.cuh).
 *
 * The ranks form nCards groups of scaleIn consecutive ranks, one group per
 * physical device (the two partitions of a GPU in DPX). A rank's rail is the
 * ranks with its partition index on every device; its siblings are the other
 * ranks of its own device. Each rank pushes its own chunk directly to every
 * other rank on its rail (one hop, not a ring) and to its siblings. A chunk
 * that arrives on the rail is forwarded to the siblings, so it crosses into
 * each device once.
 *
 * User buffers are not registered, so no peer touches recvbuff: every remote
 * write lands in the receiver's scratch at the slot of the chunk's owner, and
 * the receiver copies it from there into recvbuff.
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

// count is the per-rank chunk in elements, a multiple of 16 B. Scratch holds nRanks chunk slots, and each slot is
// written by one rank: a rank pushes its own slot to every rail peer and every sibling, and forwards each rail slot
// it receives to its siblings. The rank's own slot is never written.
//
// Block b of every rank moves the same lines of each chunk: first + k * stride for k < nRows, with
// first = b * blockDim.x + threadIdx.x and stride the grid width, so the flags pair blocks of equal index. The rows
// are cut into nPieces pieces. Piece p is posted to a rail peer as p, and to a sibling as p * nCards + g. g = 0 is
// this rank's own chunk; g >= 1 is the chunk just received from the rail peer g cards behind, so the posts one
// sender makes to one receiver only grow. Sibling piece p - 1 is drained while piece p is in flight. Every wait is
// for a post the sender made before it waited for this rank, so the exchange cannot deadlock.
template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllGatherFabricHier(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff,
                                         const T* __restrict__ sendbuff, size_t count, int selfRank, int nRanks,
                                         int scaleIn, unsigned pieces, FabricGpuBarrier barrier) {
  using FlagType = FabricGpuBarrier::FlagType;
  const size_t nLines = count * sizeof(T) / sizeof(uint4);
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Every rank derives the same piece count: it depends only on count, the grid and pieces.
  const size_t nRows = (nLines + stride - 1) / stride;
  const size_t perPiece = (nRows + pieces - 1) / pieces;
  const unsigned nPieces = static_cast<unsigned>((nRows + perPiece - 1) / perPiece);
  auto pieceOf = [&](unsigned p) -> std::pair<size_t, size_t> {
    const size_t lo = first + static_cast<size_t>(p) * perPiece * stride;
    const size_t hi = first + static_cast<size_t>(p + 1) * perPiece * stride;
    return {lo, hi < nLines ? hi : nLines};
  };

  const int nCards = nRanks / scaleIn;
  const int card = selfRank / scaleIn;
  const int local = selfRank - card * scaleIn;
  // Card `delta` steps from this one, wrapped. +g is the rail peer we push to; -g is the one we receive from.
  auto cardAt = [&](int delta) {
    int c = card + delta;
    if (c >= nCards) c -= nCards;
    else if (c < 0) c += nCards;
    return c;
  };
  // Sibling d (1 <= d < scaleIn) is the rank d partitions after this one on its device.
  auto sibling = [&](int d) {
    int l = local + d;
    if (l >= scaleIn) l -= scaleIn;
    return card * scaleIn + l;
  };

  uint4* const recvLines = reinterpret_cast<uint4*>(recvbuff);
  const uint4* const sendLines = reinterpret_cast<const uint4*>(sendbuff);
  const uint4* const selfScratch = reinterpret_cast<const uint4*>(ipcbuffs[selfRank]);
  const size_t ownSlot = static_cast<size_t>(selfRank) * nLines;
  const unsigned nCardPieces = nPieces * static_cast<unsigned>(nCards);

  // The previous collective's readers must be done with this rank's scratch before peers overwrite it.
  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  const FlagType epoch = barrier.epochBegin();

  // Stores one loaded line of `slot` into recvbuff and into every sibling's scratch.
  auto storeRecvAndSiblings = [&](size_t slot, size_t line, uint4 v) {
    recvLines[slot + line] = v;
#pragma unroll
    for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
      if (d < scaleIn) {
        uint4* dst = reinterpret_cast<uint4*>(ipcbuffs[sibling(d)]) + slot;
        dst[line] = v;
      }
    }
  };

  // Copies piece p of every sibling. g = 0 is that sibling's own chunk; g >= 1 is the chunk it forwarded from the
  // rail peer g cards behind.
  auto drainSiblings = [&](unsigned p) {
#pragma unroll
    for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
      if (d < scaleIn) {
        const int sib = sibling(d);
        const int sibLocal = sib - card * scaleIn;
        const FlagType* const flag = barrier.epochFlagFrom(sib);
        for (int g = 0; g < nCards; ++g) {
          barrier.epochWaitPiece(flag, epoch, p * static_cast<unsigned>(nCards) + static_cast<unsigned>(g));
          int dataCard = card - g;
          if (dataCard < 0) dataCard += nCards;
          const size_t slot = static_cast<size_t>(dataCard * scaleIn + sibLocal) * nLines;
          auto [lo, hi] = pieceOf(p);
          ddaAgHierScatter<true>(selfScratch + slot, lo, hi, stride, [&](size_t line, uint4 v) {
            recvLines[slot + line] = v;
          });
        }
      }
    }
  };

  for (unsigned p = 0; p <= nPieces; ++p) {
    if (p < nPieces) {
      auto [lo, hi] = pieceOf(p);
      // Own chunk: local recv, every other rail peer, and every sibling. Issued before the drain so the stores
      // travel while the previous sibling piece is copied out.
      ddaAgHierScatter<false>(sendLines, lo, hi, stride, [&](size_t line, uint4 v) {
        recvLines[ownSlot + line] = v;
        for (int g = 1; g < nCards; ++g) {
          const int peer = cardAt(g) * scaleIn + local;
          uint4* dst = reinterpret_cast<uint4*>(ipcbuffs[peer]) + ownSlot;
          dst[line] = v;
        }
#pragma unroll
        for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
          if (d < scaleIn) {
            uint4* dst = reinterpret_cast<uint4*>(ipcbuffs[sibling(d)]) + ownSlot;
            dst[line] = v;
          }
        }
      });
    }
    if (p > 0) drainSiblings(p - 1);
    if (p < nPieces) {
      barrier.orderDestWrites();
      for (int g = 1; g < nCards; ++g) {
        barrier.epochPostPieceDrained(cardAt(g) * scaleIn + local, epoch, p);
      }
#pragma unroll
      for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
        if (d < scaleIn) barrier.epochPostPieceDrained(sibling(d), epoch, p * static_cast<unsigned>(nCards));
      }

      // Rail alltoall: the peer g cards behind has pushed its own chunk here. Copy it out and forward it to the
      // siblings. That peer's post of piece p is its own-chunk post above, which runs before this wait.
      for (int g = 1; g < nCards; ++g) {
        const int src = cardAt(-g) * scaleIn + local;
        const size_t slot = static_cast<size_t>(src) * nLines;
        barrier.epochWaitPiece(barrier.epochFlagFrom(src), epoch, p);
        auto [lo, hi] = pieceOf(p);
        ddaAgHierScatter<true>(selfScratch + slot, lo, hi, stride, [&](size_t line, uint4 v) {
          storeRecvAndSiblings(slot, line, v);
        });
      }
      barrier.orderDestWrites();
#pragma unroll
      for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
        if (d < scaleIn) {
          const int sib = sibling(d);
          for (int g = 1; g < nCards; ++g) {
            barrier.epochPostPieceDrained(sib, epoch, p * static_cast<unsigned>(nCards) + static_cast<unsigned>(g));
          }
        }
      }
    }
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
  // One past the largest sibling piece (p * nCards + g), so no flag this launch posted reaches the next launch.
  barrier.epochEnd(epoch, nCardPieces + 1);
}

} // namespace dda::common
