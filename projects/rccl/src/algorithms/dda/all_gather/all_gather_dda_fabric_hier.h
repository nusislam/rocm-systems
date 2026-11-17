/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Hierarchical DDA all-gather for the fabric/VMM path, after the symmetric
 * rail-ring AllGather (agAlgoHier in src/device/symmetric/all_gather_gin.cuh).
 *
 * The ranks form nCards groups of scaleIn consecutive ranks, one group per
 * physical device (the two partitions of a GPU in DPX). A rank's rail is the
 * ranks with its partition index on every device; its siblings are the other
 * ranks of its own device. Each chunk travels its rail as a ring, and every rank
 * that owns or receives a chunk forwards it to the next rail peer and broadcasts
 * it to its siblings, so a chunk crosses into each device once.
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

// Copies 16 B lines lo, lo + stride, ... < hi of src into recv and into every non-null dst.
// SysLoad: src is scratch written by other GPUs, read with system-scope loads.
template <bool SysLoad>
__device__ __forceinline__ void ddaAgHierCopy(uint4* recv, uint4* const (&dst)[kDdaAgHierMaxScaleIn],
                                              const uint4* src, size_t lo, size_t hi, size_t stride) {
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
    for (int u = 0; u < kUnroll; ++u) recv[i + u * stride] = v[u];
#pragma unroll
    for (int d = 0; d < kDdaAgHierMaxScaleIn; ++d) {
      if (dst[d] != nullptr) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) dst[d][i + u * stride] = v[u];
      }
    }
  }
  for (; i < hi; i += stride) {
    uint4 v;
    load(i, v);
    recv[i] = v;
#pragma unroll
    for (int d = 0; d < kDdaAgHierMaxScaleIn; ++d) {
      if (dst[d] != nullptr) dst[d][i] = v;
    }
  }
}

// count is the per-rank chunk in elements, a multiple of 16 B. Scratch holds nRanks chunk slots.
//
// Block b of every rank moves the same lines of each chunk: first + k * stride for k < nRows, with
// first = b * blockDim.x + threadIdx.x and stride the grid width, so the flags pair blocks of equal index. The rows
// are cut into nPieces pieces. Ring step s carries the chunk of the rail peer on device (card - s): step 0 reads
// sendbuff, later steps read the scratch slot the previous rail peer filled. Each piece goes to recvbuff, to the
// next rail peer (all but the last step) and to every sibling, and is posted as piece s * nPieces + p, so the posts
// a receiver sees from one sender only grow. Sibling pieces are drained one iteration behind, by which time they are
// usually there. Every wait is for a post made in an earlier iteration, so the ring cannot deadlock.
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
  const int railNext = (card + 1 == nCards ? 0 : card + 1) * scaleIn + local;
  const int railPrev = (card == 0 ? nCards - 1 : card - 1) * scaleIn + local;
  // Sibling d (1 <= d < scaleIn) is the rank d partitions after this one on its device.
  auto sibling = [&](int d) {
    int l = local + d;
    if (l >= scaleIn) l -= scaleIn;
    return card * scaleIn + l;
  };

  uint4* const recvLines = reinterpret_cast<uint4*>(recvbuff);
  const uint4* const sendLines = reinterpret_cast<const uint4*>(sendbuff);
  const uint4* const selfScratch = reinterpret_cast<const uint4*>(ipcbuffs[selfRank]);
  uint4* const noDst[kDdaAgHierMaxScaleIn] = {};

  // The previous collective's readers must be done with this rank's scratch before peers overwrite it.
  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  const FlagType epoch = barrier.epochBegin();
  const FlagType* const railFlag = barrier.epochFlagFrom(railPrev);

  // Copies piece q of every sibling, which carries the chunk of that sibling's rail peer on device (card - s).
  auto drainSiblings = [&](unsigned q) {
    const int s = static_cast<int>(q / nPieces);
    int dataCard = card - s;
    if (dataCard < 0) dataCard += nCards;
    auto [lo, hi] = pieceOf(q % nPieces);
#pragma unroll
    for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
      if (d < scaleIn) {
        const int sib = sibling(d);
        barrier.epochWaitPiece(barrier.epochFlagFrom(sib), epoch, q);
        const size_t slot = static_cast<size_t>(dataCard * scaleIn + (sib - card * scaleIn)) * nLines;
        ddaAgHierCopy<true>(recvLines + slot, noDst, selfScratch + slot, lo, hi, stride);
      }
    }
  };

  for (int s = 0; s < nCards; ++s) {
    int dataCard = card - s;
    if (dataCard < 0) dataCard += nCards;
    const size_t slot = static_cast<size_t>(dataCard * scaleIn + local) * nLines;
    uint4* dst[kDdaAgHierMaxScaleIn];
    dst[0] = s + 1 < nCards ? reinterpret_cast<uint4*>(ipcbuffs[railNext]) + slot : nullptr;
#pragma unroll
    for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
      dst[d] = d < scaleIn ? reinterpret_cast<uint4*>(ipcbuffs[sibling(d)]) + slot : nullptr;
    }

    for (unsigned p = 0; p < nPieces; ++p) {
      const unsigned q = static_cast<unsigned>(s) * nPieces + p;
      auto [lo, hi] = pieceOf(p);
      if (s == 0) {
        ddaAgHierCopy<false>(recvLines + slot, dst, sendLines, lo, hi, stride);
      } else {
        barrier.epochWaitPiece(railFlag, epoch, q - nPieces);
        ddaAgHierCopy<true>(recvLines + slot, dst, selfScratch + slot, lo, hi, stride);
      }

      barrier.orderDestWrites();
      if (s + 1 < nCards) barrier.epochPostPieceDrained(railNext, epoch, q);
#pragma unroll
      for (int d = 1; d < kDdaAgHierMaxScaleIn; ++d) {
        if (d < scaleIn) barrier.epochPostPieceDrained(sibling(d), epoch, q);
      }

      if (q > 0) drainSiblings(q - 1);
    }
  }
  drainSiblings(static_cast<unsigned>(nCards) * nPieces - 1);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
  // One past the largest piece, so no flag this launch posted reaches the next launch's epoch.
  barrier.epochEnd(epoch, static_cast<unsigned>(nCards) * nPieces + 1);
}

} // namespace dda::common
