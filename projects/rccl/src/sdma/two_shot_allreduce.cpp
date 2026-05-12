/*************************************************************************
 * Anvil SDMA two-shot AllReduce: reduce-scatter + allgather via
 * putSignal into each rank's sdmaTempBuffer, then copy to recvbuff.
 ************************************************************************/

#include "checks.h"
#include "comm.h"
#include "core.h"
#include "debug.h"
#include "nccl.h"
#include "sdma/anvil_device.hpp"
#include "sdma/two_shot_allreduce_launch.hpp"

#include <climits>
#include <hip/hip_runtime.h>

static constexpr int kNumSdmaChannels = 2;

static __host__ inline size_t crossHeaderBytes(int nRanks) {
  size_t raw = 2ull * (size_t)nRanks * sizeof(uint64_t);
  return (raw + 255ull) & ~((size_t)255);
}

RCCL_PARAM(AnvilTwoShotAllreduce, "ANVIL_TWO_SHOT_ALLREDUCE", 0);

__global__ void anvilTwoShotPhase1Kernel(const float* __restrict__ sendbuff, float* __restrict__ recvbuff,
                                         int count, int nRanks, int myRank, void** __restrict__ peerTempPtrs,
                                         rocshmem::anvil::SdmaQueueDeviceHandle** __restrict__ devHandles,
                                         size_t dataByteOffset) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  const int chunk = count / nRanks;
  const size_t chunkBytes = (size_t)chunk * sizeof(float);
  char* myTemp = reinterpret_cast<char*>(peerTempPtrs[myRank]);
  float* stage1 = reinterpret_cast<float*>(myTemp + dataByteOffset);

  for (int peer = 0; peer < nRanks; ++peer) {
    if (peer == myRank)
      continue;
    void* peerBase = peerTempPtrs[peer];
    char* peerC = reinterpret_cast<char*>(peerBase);
    float* dst = reinterpret_cast<float*>(peerC + dataByteOffset) + (size_t)myRank * (size_t)chunk;
    const float* src = sendbuff + (size_t)myRank * (size_t)chunk;
    uint64_t* sig = reinterpret_cast<uint64_t*>(peerBase) + myRank;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr == nullptr)
      return;
    rocshmem::anvil::SdmaQueueDeviceHandle& hq = *hqPtr;
    rocshmem::anvil::putSignal(hq, dst, const_cast<float*>(src), chunkBytes, sig);
  }

  for (int s = 0; s < nRanks; ++s) {
    if (s == myRank)
      continue;
    uint64_t* waitAt = reinterpret_cast<uint64_t*>(myTemp) + s;
    rocshmem::anvil::waitSignal(waitAt, 1ull);
  }

  for (int i = 0; i < chunk; ++i) {
    const size_t idx = (size_t)myRank * (size_t)chunk + (size_t)i;
    float acc = sendbuff[idx];
    for (int s = 0; s < nRanks; ++s) {
      if (s == myRank)
        continue;
      acc += stage1[(size_t)s * (size_t)chunk + (size_t)i];
    }
    recvbuff[idx] = acc;
  }
}

__global__ void anvilTwoShotPhase2Kernel(const float* __restrict__ sendbuff, float* __restrict__ recvbuff,
                                         int count, int nRanks, int myRank, void** __restrict__ peerTempPtrs,
                                         rocshmem::anvil::SdmaQueueDeviceHandle** __restrict__ devHandles,
                                         size_t dataByteOffset) {
  (void)sendbuff;
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  const int chunk = count / nRanks;
  const size_t chunkBytes = (size_t)chunk * sizeof(float);
  const size_t stageBytes = (size_t)nRanks * chunkBytes;
  char* myTemp = reinterpret_cast<char*>(peerTempPtrs[myRank]);
  const size_t stage2Off = dataByteOffset + stageBytes;

  for (int peer = 0; peer < nRanks; ++peer) {
    if (peer == myRank)
      continue;
    void* peerBase = peerTempPtrs[peer];
    char* peerC = reinterpret_cast<char*>(peerBase);
    float* dst = reinterpret_cast<float*>(peerC + stage2Off) + (size_t)myRank * (size_t)chunk;
    const float* src = recvbuff + (size_t)myRank * (size_t)chunk;
    uint64_t* sig = reinterpret_cast<uint64_t*>(peerBase) + nRanks + myRank;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 1];
    if (hqPtr == nullptr)
      hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr == nullptr)
      return;
    rocshmem::anvil::SdmaQueueDeviceHandle& hq = *hqPtr;
    rocshmem::anvil::putSignal(hq, dst, const_cast<float*>(src), chunkBytes, sig);
  }

  for (int s = 0; s < nRanks; ++s) {
    if (s == myRank)
      continue;
    uint64_t* waitAt = reinterpret_cast<uint64_t*>(myTemp) + nRanks + s;
    rocshmem::anvil::waitSignal(waitAt, 1ull);
  }

  float* myStage2 = reinterpret_cast<float*>(myTemp + stage2Off);
  for (int s = 0; s < nRanks; ++s) {
    if (s == myRank)
      continue;
    const float* src = myStage2 + (size_t)s * (size_t)chunk;
    float* dst = recvbuff + (size_t)s * (size_t)chunk;
    for (int i = 0; i < chunk; ++i)
      dst[i] = src[i];
  }
}

ncclResult_t rcclAnvilTwoShotAllReduceTry(const void* sendbuff, void* recvbuff, size_t count,
                                          ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                          cudaStream_t stream) {
  if (rcclParamAnvilTwoShotAllreduce() == 0)
    return ncclInvalidUsage;
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr)
    return ncclInvalidArgument;
  if (datatype != ncclFloat || op != ncclSum)
    return ncclInvalidUsage;
  if (comm->nNodes != 1)
    return ncclInvalidUsage;

  const int nr = comm->nRanks;
  if (nr <= 0 || comm->sdmaTempBuffer == nullptr || comm->sdmaTempBufferPeerPtrs_d == nullptr ||
      comm->deviceHandles_d == nullptr)
    return ncclInvalidUsage;

  if (count > (size_t)INT_MAX || (int)count % nr != 0)
    return ncclInvalidUsage;

  const int icount = (int)count;
  const int chunk = icount / nr;
  const size_t chunkBytes = (size_t)chunk * sizeof(float);
  const size_t stageBytes = (size_t)nr * chunkBytes;
  const size_t hdrExpected = crossHeaderBytes(nr);
  if (comm->sdmaTempBufferCrossSignalBytes != hdrExpected ||
      comm->sdmaTempBufferDataByteOffset != hdrExpected)
    return ncclInvalidUsage;

  const size_t need = comm->sdmaTempBufferDataByteOffset + 2ull * stageBytes;
  if (need > comm->sdmaTempBufferBytes)
    return ncclInvalidUsage;

  const float* sb = reinterpret_cast<const float*>(sendbuff);
  float* rb = reinterpret_cast<float*>(recvbuff);

  CUDACHECK(hipMemsetAsync(comm->sdmaTempBuffer, 0, comm->sdmaTempBufferCrossSignalBytes, stream));

  hipLaunchKernelGGL(anvilTwoShotPhase1Kernel, dim3(1), dim3(1), 0, stream, sb, rb, icount, nr,
                     comm->rank, comm->sdmaTempBufferPeerPtrs_d, comm->deviceHandles_d,
                     comm->sdmaTempBufferDataByteOffset);
  CUDACHECK(hipGetLastError());

  CUDACHECK(hipMemsetAsync(comm->sdmaTempBuffer, 0, comm->sdmaTempBufferCrossSignalBytes, stream));

  hipLaunchKernelGGL(anvilTwoShotPhase2Kernel, dim3(1), dim3(1), 0, stream, sb, rb, icount, nr, comm->rank,
                     comm->sdmaTempBufferPeerPtrs_d, comm->deviceHandles_d,
                     comm->sdmaTempBufferDataByteOffset);
  CUDACHECK(hipGetLastError());

  return ncclSuccess;
}
