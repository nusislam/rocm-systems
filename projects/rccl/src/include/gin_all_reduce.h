/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * GIN tree all-reduce (remote-write / put based), launched from ncclAllReduce.
 ************************************************************************/

#ifndef GIN_ALL_REDUCE_H_
#define GIN_ALL_REDUCE_H_

#include "nccl.h"

struct ncclComm;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
bool ncclGinAllReduceSdmaBackendEnabled(struct ncclComm* comm);
bool ncclAllReduceGinTreeEligible(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op);
ncclResult_t ncclAllReduceGinTree(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, struct ncclComm* comm, hipStream_t stream);
ncclResult_t ncclGinAllReduceInitOnce(struct ncclComm* comm);
ncclResult_t ncclGinAllReduceFinalize(struct ncclComm* comm);
#endif

#endif
