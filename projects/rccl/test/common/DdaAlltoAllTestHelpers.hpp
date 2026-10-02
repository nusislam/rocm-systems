/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstring>

#include "archinfo.h"
#include "collectives.h"
#include "comm.h"
#include "enqueue.h"
#include "group.h"
#include "algorithms/dda/dda_init_detail.h"
#include "rccl_common.h"

namespace RcclUnitTesting
{

// Use the production rcclDdaEnabled() from rccl_common.h directly.

inline size_t testAlltoAllTotalBytes(size_t count, int nRanks, ncclDataType_t datatype) {
  return static_cast<size_t>(nRanks) * count * static_cast<size_t>(ncclTypeSize(datatype));
}

inline size_t testDdaAlltoAllThreshold(const ncclComm* comm) {
  return rcclDdaEntryThreshold(comm, ncclFuncAlltoAll);
}

inline bool testRcclDdaAlltoAllThresholdEnabled(
    const ncclComm* comm,
    size_t count,
    ncclDataType_t datatype) {
  return rcclDdaEnabled(
      comm,
      testAlltoAllTotalBytes(count, comm->nRanks, datatype),
      testDdaAlltoAllThreshold(comm));
}

// IPC AllToAll copies sendbuff into scratch inside the kernel at every size.
// The fabric path still stages via a pre-kernel memcpy.
inline bool testAlltoAllUsesInKernelStagingCopy(size_t countPerRank, ncclDataType_t datatype) {
  (void)countPerRank;
  (void)datatype;
  return true;
}

inline size_t testAlltoAllDdaIpcStagingBytes(size_t count, int nRanks, size_t typeSize) {
  return count * static_cast<size_t>(nRanks) * typeSize;
}

struct DdaAlltoAllMockComm
{
    ncclComm comm{};
    char archNameBuf[64]{};

    DdaAlltoAllMockComm(const DdaAlltoAllMockComm&)            = delete;
    DdaAlltoAllMockComm& operator=(const DdaAlltoAllMockComm&) = delete;

    DdaAlltoAllMockComm() { reset("gfx950:sramecc+:xnack-"); }

    void reset(const char* archName)
    {
        std::memset(&comm, 0, sizeof(comm));
        std::strncpy(archNameBuf, archName, sizeof(archNameBuf) - 1);
        archNameBuf[sizeof(archNameBuf) - 1] = '\0';
        comm.archName = archNameBuf;
        comm.nNodes = 1;
        comm.nRanks = nccl_dda_detail::kDdaNranks;
        comm.symmetricSupport = 0;
    }

    ncclComm* get() { return &comm; }
};

// Largest float32 per-rank count whose 8-rank AlltoAll totals exactly 4 MiB.
constexpr size_t kAlltoAllFloat32CountAt4MbThreshold =
    4194304UL /
    (static_cast<size_t>(nccl_dda_detail::kDdaNranks) * sizeof(float));

// Per-rank float32 count whose 8-rank AlltoAll totals exactly 1 MiB (gfx1250 LL128 ceiling).
constexpr size_t kAlltoAllFloat32CountAt1MbLL128Threshold =
    1048576UL /
    (static_cast<size_t>(nccl_dda_detail::kDdaNranks) * sizeof(float));

// 4 KiB/rank float32: single-block grid on an 8-rank IPC launch.
constexpr size_t kAlltoAllFloat32CountAt4KbPerRank = 1024;

// 8 KiB/rank float32: multi-block grid. IPC still copies inside the kernel.
constexpr size_t kAlltoAllFloat32CountAt8KbPerRank = 2048;

} // namespace RcclUnitTesting
