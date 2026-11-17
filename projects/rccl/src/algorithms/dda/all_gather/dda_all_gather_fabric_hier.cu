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
// RCCL_DDA_AG_HIER_PIECES: pieces each chunk is cut into to pipeline the ring. 0 (default) = one per 512 KiB of the
// chunk, 1 to 16; > 0 requests that many, capped at 64 and at the rows each thread copies. Must be the same on every
// rank.
RCCL_PARAM(DdaAgHierPieces, "DDA_AG_HIER_PIECES", 0);

namespace {

using dda::common::kDdaAgHierMaxScaleIn;
using nccl_dda_detail::DdaFabricBarrierState;

constexpr uint32_t kDdaAgHierThreads = 512;
constexpr size_t kDdaAgHierDefaultBlocks = 64;
constexpr unsigned kDdaAgHierMaxAutoPieces = 16;

// Every rank gets the same grid: it depends only on the chunk size, the env and the init-time block cap.
static inline std::pair<dim3, dim3> ddaAllGatherFabricHierGeom(ncclComm* comm, size_t bytes) {
  const int64_t forced = rcclParamDdaAgHierBlocks();
  size_t blocks = forced > 0 ? static_cast<size_t>(forced) : kDdaAgHierDefaultBlocks;
  // No block without a full row of lines to copy.
  blocks = std::min(blocks, std::max<size_t>(1, bytes / sizeof(uint4) / kDdaAgHierThreads));
  blocks = std::min(blocks, static_cast<size_t>(std::max(comm->ddaFabricMaxBlocks, 1)));
  return {dim3(static_cast<unsigned>(blocks)), dim3(kDdaAgHierThreads)};
}

static inline unsigned ddaAllGatherFabricHierPieces(size_t bytes) {
  const int64_t forced = rcclParamDdaAgHierPieces();
  if (forced > 0) return static_cast<unsigned>(std::min<int64_t>(forced, 64));
  return static_cast<unsigned>(std::clamp<size_t>(bytes >> 19, 1, kDdaAgHierMaxAutoPieces));
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
  if (bytes * nRanks > comm->ddaScratchBytes) {
    WARN("DDA fabric hierarchical allgather: %zu bytes per rank need %zu bytes; comm scratch is %zu bytes", bytes,
         bytes * nRanks, comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  const auto gridBlock = ddaAllGatherFabricHierGeom(comm, bytes);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;
  const unsigned pieces = ddaAllGatherFabricHierPieces(bytes);
  const int scaleIn = comm->ddaHierScaleIn;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;
  int8_t** d_ipcbuffs = reinterpret_cast<int8_t**>(comm->ddaPeerPtrsDev);

  INFO(NCCL_COLL,
       "DDA fabric hierarchical AllGather: launching kernel: nRanks=%d devices=%d ranksPerDevice=%d bytes=%zu "
       "grid=%u block=%u pieces=%u",
       nRanks, nRanks / scaleIn, scaleIn, bytes, grid.x, block.x, pieces);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricHier<int8_t>), grid, block, 0, stream,
                        /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<int8_t*>(recvbuff),
                        static_cast<const int8_t*>(sendbuff), bytes, comm->rank, nRanks, scaleIn, pieces,
                        barrierHost);
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
  // Every chunk has a scratch slot, since no peer can write recvbuff.
  if (bytes * comm->nRanks > comm->ddaScratchBytes) {
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
