/*************************************************************************
 * Optional Anvil SDMA two-shot AllReduce (float sum, single-node).
 ************************************************************************/
#pragma once

#include "nccl.h"

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <stdint.h>
int64_t rcclParamAnvilAlltoAll(void);
#endif

// Returns ncclSuccess when launched, ncclInvalidUsage to fall back to generic path,
// ncclInvalidArgument for invalid pointers, ncclInternalError on unexpected HIP failures.
ncclResult_t rcclAnvilAlltoAllTry(const void* sendbuff, void* recvbuff, size_t count,
                                          ncclDataType_t datatype, ncclComm_t comm,
                                          hipStream_t stream, uint64_t bar1, uint64_t bar2, uint64_t signal1, uint64_t bar3);

