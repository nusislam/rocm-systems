/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_gather/dda_all_gather.h"

#include "algorithms/dda/all_gather/all_gather_dda_fabric_hier.h"
#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

// RCCL_DDA_AG_HIER: 1 (default) lets the selector take the hierarchical fabric AllGather inside the arch table's
// ddaAgHier window; 0 turns it off.
RCCL_PARAM(DdaAgHier, "DDA_AG_HIER", 1);
// RCCL_DDA_AG_HIER_BLOCKS: grid size, at most the init block cap. 0 (default) = kDdaAgHierDefaultBlocks.
// Must be the same on every rank.
RCCL_PARAM(DdaAgHierBlocks, "DDA_AG_HIER_BLOCKS", 0);
// RCCL_DDA_AG_HIER_CHUNK_BYTES: largest chunk a block moves between flags. 0 (default) = kDdaAgHierDefaultChunkBytes;
// otherwise rounded down to 16 B, at least 16 B. Must be the same on every rank.
RCCL_PARAM(DdaAgHierChunkBytes, "DDA_AG_HIER_CHUNK_BYTES", 0);

namespace {

using dda::common::kDdaAgHierMaxScaleIn;
using nccl_dda_detail::DdaFabricBarrierState;

constexpr uint32_t kDdaAgHierThreads = 512;
constexpr size_t kDdaAgHierDefaultBlocks = 64;
// Matches ncclSymkAgHierMaxChunkBytes of the symmetric hierarchical AllGather.
constexpr size_t kDdaAgHierDefaultChunkBytes = 256 << 10;

// Every rank gets the same grid: it depends only on the chunk size, the env and the init-time block cap.
static inline std::pair<dim3, dim3> ddaAllGatherFabricHierGeom(ncclComm* comm, size_t bytes) {
  const int64_t forced = rcclParamDdaAgHierBlocks();
  size_t blocks = forced > 0 ? static_cast<size_t>(forced) : kDdaAgHierDefaultBlocks;
  // No block without a full row of lines to copy.
  blocks = std::min(blocks, std::max<size_t>(1, bytes / sizeof(uint4) / kDdaAgHierThreads));
  blocks = std::min(blocks, static_cast<size_t>(std::max(comm->ddaFabricMaxBlocks, 1)));
  return {dim3(static_cast<unsigned>(blocks)), dim3(kDdaAgHierThreads)};
}

// Largest chunk in 16 B lines.
static inline size_t ddaAllGatherFabricHierChunkLines() {
  const int64_t forced = rcclParamDdaAgHierChunkBytes();
  const size_t bytes = forced > 0 ? static_cast<size_t>(forced) : kDdaAgHierDefaultChunkBytes;
  return std::max<size_t>(1, bytes / sizeof(uint4));
}

// Lines of each rank's slot per round: the whole chunk when every rank's chunk fits the scratch (one round), else
// what fits half of it (double-buffered rounds). 0 when the scratch cannot hold one line per rank per half.
static inline size_t ddaAllGatherFabricHierWindowLines(const ncclComm* comm, size_t bytes) {
  const size_t nLines = bytes / sizeof(uint4);
  const size_t nRanks = static_cast<size_t>(comm->nRanks);
  const size_t scratchLines = comm->ddaScratchBytes / sizeof(uint4);
  if (nLines <= scratchLines / nRanks) return nLines;
  return std::min(nLines, scratchLines / (2 * nRanks));
}

// The comm splits into two or more devices of scaleIn consecutive ranks each, and the kernel handles scaleIn.
static bool ddaAllGatherFabricHierTopologyOk(const ncclComm* comm) {
  const int scaleIn = comm->ddaHierScaleIn;
  return scaleIn >= 2 && scaleIn <= kDdaAgHierMaxScaleIn && comm->nRanks % scaleIn == 0 &&
         comm->nRanks / scaleIn >= 2;
}

static ncclResult_t ncclAllGatherDdaFabricHierBytes(const void* sendbuff, void* recvbuff, size_t bytes,
                                                    ncclComm* comm, cudaStream_t stream) {
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaFabricBarrierState == nullptr || !ddaAllGatherFabricHierTopologyOk(comm)) {
    return ncclInvalidUsage;
  }

  const int nRanks = comm->nRanks;
  const size_t windowLines = ddaAllGatherFabricHierWindowLines(comm, bytes);
  if (windowLines == 0) {
    WARN("DDA fabric hierarchical allgather: comm scratch of %zu bytes cannot hold two 16-byte lines per rank",
         comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  const auto gridBlock = ddaAllGatherFabricHierGeom(comm, bytes);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;
  const size_t chunkLines = ddaAllGatherFabricHierChunkLines();
  const int scaleIn = comm->ddaHierScaleIn;
  const size_t nLines = bytes / sizeof(uint4);
  const size_t nRounds = (nLines + windowLines - 1) / windowLines;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;
  int8_t** d_ipcbuffs = reinterpret_cast<int8_t**>(comm->ddaPeerPtrsDev);

  INFO(NCCL_COLL,
       "DDA fabric hierarchical AllGather: launching kernel: nRanks=%d devices=%d ranksPerDevice=%d bytes=%zu "
       "grid=%u block=%u group0Blocks=%d groupBlocks=%d maxChunkBytes=%zu rounds=%zu windowBytes=%zu",
       nRanks, nRanks / scaleIn, scaleIn, bytes, grid.x, block.x,
       dda::common::ddaAgHierG0Blocks(static_cast<int>(grid.x), nRanks / scaleIn),
       dda::common::ddaAgHierGroupBlocks(static_cast<int>(grid.x), nRanks / scaleIn), chunkLines * sizeof(uint4),
       nRounds, windowLines * sizeof(uint4));

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricHier<int8_t>), grid, block, 0, stream,
                        /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<int8_t*>(recvbuff),
                        static_cast<const int8_t*>(sendbuff), bytes, comm->rank, nRanks, scaleIn, chunkLines,
                        windowLines, barrierHost);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaFabricHierEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                        ncclDataType_t datatype) {
  if (!rcclParamDdaAgHier()) {
    return false;
  }
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaFabricBarrierState == nullptr) {
    return false;
  }
  if (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks || !ddaAllGatherFabricHierTopologyOk(comm)) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  const size_t bytes = sendcount * ncclTypeSize(datatype);
  // Every chunk goes through a scratch slot, since no peer can write recvbuff; a chunk too large for the scratch
  // goes in rounds, so only a scratch without room for one line per rank per half is too small.
  if (ddaAllGatherFabricHierWindowLines(comm, bytes) == 0) {
    return false;
  }
  // The kernel moves 16-byte lines of both user buffers.
  if (bytes % 16 != 0 || !dda::common::ddaUserBuffers16ByteAligned(sendbuff, recvbuff)) {
    return false;
  }
  return true;
}

uint32_t ncclAllGatherDdaFabricHierBlocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype) {
  const auto grid = ddaAllGatherFabricHierGeom(comm, sendcount * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllGatherDdaFabricHier(const void* sendbuff, void* recvbuff, size_t sendcount,
                                        ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return ncclInvalidArgument;
  }
  return ncclAllGatherDdaFabricHierBytes(sendbuff, recvbuff, sendcount * ncclTypeSize(datatype), comm, stream);
}
