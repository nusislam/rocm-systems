/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the GIN-SDMA AllReduce path (LSA one-shot <= 16 MiB,
 * LSA two-shot > 16 MiB) launched from ncclAllReduce when symmetric windows are used.
 * See LICENSE.txt for license information.
 ******************************************************************************/

#ifndef GIN_ALL_REDUCE_H_
#define GIN_ALL_REDUCE_H_

#include "nccl.h"
#include "nccl_device.h"

struct ncclComm;

// LSA one-shot for messages <= kGinAllReduceLsaOneShotMaxBytes; two-shot above that.
constexpr int kGinAllReduceLsaCtas = 56;
constexpr int kGinAllReduceLsaThreadsPerCta = 512;
constexpr size_t kGinAllReduceLsaOneShotMaxBytes = 16ULL * 1024 * 1024;

// Lazily created on the first eligible AllReduce and torn down with the comm.
// Declared unconditionally: ncclComm embeds this even when ENABLE_ROCSHMEM_GIN is off.
struct ncclGinAllReduceState {
  bool initialized;
  struct ncclDevComm devComm;
  uint64_t* reduceDoneSync; // device sync counter for the LSA two-shot kernel
};

#if defined(ENABLE_ROCSHMEM_GIN)

bool ncclAllReduceGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclGinAllReduceFinalize(ncclComm* comm);

#else

inline ncclResult_t ncclGinAllReduceFinalize(ncclComm*) { return ncclSuccess; }

#endif

#endif
