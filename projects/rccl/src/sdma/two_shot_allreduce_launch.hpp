/*************************************************************************
 * Optional Anvil SDMA two-shot AllReduce (float sum, single-node).
 ************************************************************************/
#pragma once

#include "nccl.h"

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <stdint.h>
// RCCL_ANVIL_TWO_SHOT_ALLREDUCE — non-zero enables the path from ncclAllReduce_impl (default 0).
int64_t rcclParamAnvilTwoShotAllreduce(void);
#endif

// Returns ncclSuccess when launched, ncclInvalidUsage to fall back to generic path,
// ncclInvalidArgument for invalid pointers, ncclInternalError on unexpected HIP failures.
ncclResult_t rcclAnvilTwoShotAllReduceTry(const void* sendbuff, void* recvbuff, size_t count,
                                          ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                          hipStream_t stream);

