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
					float* __restrict__ tmpbuff, int count, int nRanks, int myRank, 
					void** __restrict__ peerTempPtrs,
                                         rocshmem::anvil::SdmaQueueDeviceHandle** __restrict__ devHandles,
                                         uint64_t** __restrict__ remoteSignals, uint64_t** remoteBarriers, 
					 uint64_t* localSignals, uint64_t* localBarriers ) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  const int chunk = count / nRanks;
  const size_t chunkBytes = (size_t)chunk * sizeof(float);
  char* myTemp = reinterpret_cast<char*>(peerTempPtrs[myRank]);
  //float* stage1 = reinterpret_cast<float*>(myTemp + dataByteOffset);

  for (int peer = 0; peer < nRanks; ++peer) {
    if (peer == myRank)
      continue;
    void* peerBase = peerTempPtrs[peer];
    char* peerC = reinterpret_cast<char*>(peerBase);
    float* dst = reinterpret_cast<float*>(peerC) + (size_t)myRank * (size_t)chunk;
    const float* src = sendbuff + (size_t)peer * (size_t)chunk;

    //uint64_t* sigPeer = remoteSignals[peer];
    uint64_t* sig = remoteSignals[peer] + myRank;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr == nullptr)
      return;
    rocshmem::anvil::SdmaQueueDeviceHandle& hq = *hqPtr;
    rocshmem::anvil::putSignal(hq, dst, const_cast<float*>(src), chunkBytes, sig);
  }

  for (int peer = 0; peer < nRanks; ++peer) {
    if (peer == myRank)
      continue;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr == nullptr)
      return;
    rocshmem::anvil::quiet(*hqPtr);
  }

  for (int s = 0; s < nRanks; ++s) {
    if (s == myRank)
      continue;
    uint64_t* waitAt = reinterpret_cast<uint64_t*>(localSignals) + s;
    rocshmem::anvil::waitSignal(waitAt, 1ull);
  }

  for (int i = 0; i < chunk; ++i) {
    const size_t idx = (size_t)myRank * (size_t)chunk + (size_t)i;
    float acc = sendbuff[idx];

    for (int s = 0; s < nRanks; ++s) {
      if (s == myRank)
        continue;
      size_t idx1 = (size_t)s * (size_t)chunk + (size_t)i;
      float acc1 = tmpbuff[idx1];
      acc += acc1;
    }
    recvbuff[idx] = acc;
  }

  __syncthreads();

  //need a barrier
  // allgather
  //
  /*for (int peer = 0; peer < nRanks; ++peer) {
    if (peer == myRank)
      continue;
    void* peerBase = peerTempPtrs[peer];
    char* peerC = reinterpret_cast<char*>(peerBase);
    float* dst = reinterpret_cast<float*>(peerC) + (size_t)myRank * (size_t)chunk;
    const float* src = recvbuff + (size_t)myRank * (size_t)chunk;

    uint64_t* sigPeer = remoteSignals[peer];
    uint64_t* sig = sigPeer + myRank;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr == nullptr)
      return;
    rocshmem::anvil::SdmaQueueDeviceHandle& hq = *hqPtr;
    rocshmem::anvil::putSignal(hq, dst, const_cast<float*>(src), chunkBytes, sig);
  }

  for (int s = 0; s < nRanks; ++s) {
    if (s == myRank)
      continue;
    uint64_t* waitAt = reinterpret_cast<uint64_t*>(localSignals) + s;
    rocshmem::anvil::waitSignal(waitAt, 1ull);
  }

  for (int s = 0; s < nRanks; ++s) {
    if (s == myRank)
      continue;
    const float* src =  tmpbuff + (size_t)s * (size_t)chunk;
    float* dst = recvbuff + (size_t)s * (size_t)chunk;
    for (int i = 0; i < chunk; ++i)
      dst[i] = src[i];
  }*/

}

ncclResult_t rcclAnvilTwoShotAllReduceTry(const void* sendbuff, void* recvbuff, size_t count,
                                          ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                          hipStream_t stream) {
  if (rcclParamAnvilTwoShotAllreduce() == 0)
    return ncclInvalidUsage;
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr)
    return ncclInvalidArgument;
  if (datatype != ncclFloat || op != ncclSum)
    return ncclInvalidUsage;
  if (comm->nNodes != 1)
    return ncclInvalidUsage;

  const int nr = comm->nRanks;
  if (nr <= 0 || comm->sdmaFineGrainedTempBuf == nullptr || comm->sdmaFineGrainedTempPeerPtrs_d == nullptr ||
      comm->sdmaSyncBuffer == nullptr || comm->sdmaSyncBufferPeerPtrs_d == nullptr || comm->deviceHandles_d == nullptr)
    return ncclInvalidUsage;

  if (count > (size_t)INT_MAX || (int)count % nr != 0)
    return ncclInvalidUsage;

  const int icount = (int)count;
  const int chunk = icount / nr;
  const size_t chunkBytes = (size_t)chunk * sizeof(float);
  const size_t stageBytes = (size_t)nr * chunkBytes;

  //uint64_t **remoteSignals = comm->remoteSignals;
  uint64_t **remoteBarriers = comm->remoteBarriers;
  uint64_t *localBarriers = comm->localBarriers;
  //uint64_t *localSignals = comm->localSignals;


  const float* sb = reinterpret_cast<const float*>(sendbuff);
  float* rb = reinterpret_cast<float*>(recvbuff);

  CUDACHECK(hipMemsetAsync(comm->sdmaFineGrainedTempBuf, 0, comm->sdmaFineGrainedTempBytes, stream));
  CUDACHECK(hipMemsetAsync(comm->sdmaSyncBuffer, 0, comm->sdmaSyncBufferBytes, stream));

  printf("Invoking the kernel %d\n", comm->rank);
  float* tmpBuf = reinterpret_cast<float*>(comm->sdmaFineGrainedTempBuf);
  hipLaunchKernelGGL(anvilTwoShotPhase1Kernel, dim3(1), dim3(1), 0, stream, sb, rb, tmpBuf, icount, nr,
                     comm->rank, comm->sdmaFineGrainedTempPeerPtrs_d, comm->deviceHandles_d, comm->sdmaSyncBufferPeerPtrs_d, 
		     nullptr /*remoteBarriers*/, comm->sdmaSyncBuffer, nullptr /*localBarriers*/);
  CUDACHECK(hipGetLastError());


  return ncclSuccess;
}
