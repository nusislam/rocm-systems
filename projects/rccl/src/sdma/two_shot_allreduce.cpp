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

static constexpr int kNumSdmaChannels = 1;
static constexpr int kMaxSdmaBarrierBlocks = 32;

static __host__ inline size_t crossHeaderBytes(int nRanks) {
  size_t raw = 2ull * (size_t)nRanks * sizeof(uint64_t);
  return (raw + 255ull) & ~((size_t)255);
}

RCCL_PARAM(AnvilTwoShotAllreduce, "ANVIL_TWO_SHOT_ALLREDUCE", 0);

// Cross-rank barrier for threadblock `blockId` on `myRank`, using sdmaBarrierBuffer layout:
// slot = blockId * nRanks + rank (see initSdmaTempBufferIpc in init.cc).
__device__ __forceinline__ void anvilCrossRankBlockBarrier(int nRanks, int myRank, int blockId,
                                                             uint64_t** remoteBarriers,
                                                             uint64_t* localBarriers, uint64_t val) {
  const int slot = blockId * nRanks + myRank;
  uint64_t kArrived = val;


  if (threadIdx.x < nRanks) {
    //printf("val = %zu\n", val);	  
    //for (int peer = 0; peer < nRanks; ++peer) {
      int peer = threadIdx.x;
      if (peer != myRank) {
        //continue;
      	uint64_t* p = remoteBarriers[peer] + slot;
      	__atomic_store_n(p, val, __ATOMIC_RELEASE);
    //}
    //for (int peer = 0; peer < nRanks; ++peer) {
      //if (peer == myRank)
        //continue;
      	uint64_t* waitAt = localBarriers + blockId * nRanks + peer;
      	while (__atomic_load_n(waitAt, __ATOMIC_ACQUIRE) < kArrived) {
      	}
      }
    //}
  }
  __syncthreads();
}


__global__ void anvilTwoShotPhase1Kernel(const float* __restrict__ sendbuff, float* __restrict__ recvbuff, 
					float* __restrict__ tmpbuff, int count, int nRanks, int myRank, 
					void** __restrict__ peerTempPtrs,
                                         rocshmem::anvil::SdmaQueueDeviceHandle** __restrict__ devHandles,
                                         uint64_t** __restrict__ remoteSignals, uint64_t** remoteBarriers, 
					 uint64_t* localSignals, uint64_t* localBarriers, uint64_t val, uint64_t val1 ) {
  /*if (threadIdx.x != 0 || blockIdx.x != 0)
    return;*/
  if (blockIdx.x >= kMaxSdmaBarrierBlocks)
    return;

  // Barrier values 1/2/3: sdmaBarrierBuffer is reset to zero before each kernel launch.
  constexpr uint64_t kBarrierVal0 = 1ull;
  constexpr uint64_t kSignalRs = 1ull;
  constexpr uint64_t kSignalAg = 2ull;

  anvilCrossRankBlockBarrier(nRanks, myRank, blockIdx.x, remoteBarriers, localBarriers, kBarrierVal0);
  if (threadIdx.x == 0) 
	  printf("Back from barrier\n");

  const int chunk = count / nRanks;
  const size_t chunkBytes = (size_t)chunk * sizeof(float);
  char* myTemp = reinterpret_cast<char*>(peerTempPtrs[myRank]);
  int peer, s;
  //float* stage1 = reinterpret_cast<float*>(myTemp + dataByteOffset);

  //for (int peer = 0; peer < nRanks; ++peer) {
  //
  if (threadIdx.x < nRanks) {
    peer = threadIdx.x;	 
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr == nullptr)
      return;
    rocshmem::anvil::SdmaQueueDeviceHandle& hq = *hqPtr;

    if (peer != myRank) {
    	void* peerBase = peerTempPtrs[peer];
    	char* peerC = reinterpret_cast<char*>(peerBase);
    	float* dst = reinterpret_cast<float*>(peerC) + (size_t)myRank * (size_t)chunk;
    	const float* src = sendbuff + (size_t)peer * (size_t)chunk;

    //uint64_t* sigPeer = remoteSignals[peer];
    	uint64_t* sig = remoteSignals[peer] + myRank;
    	rocshmem::anvil::putSignal(hq, dst, const_cast<float*>(src), chunkBytes, sig);
    } else {
	float* dst = tmpbuff + (size_t)myRank * (size_t)chunk;
        const float* src = sendbuff + (size_t)peer * (size_t)chunk;
        rocshmem::anvil::putSelf(hq, dst, const_cast<float*>(src), chunkBytes);
    }
  }

  __syncthreads();
  //for (int peer = 0; peer < nRanks; ++peer) {
  if (threadIdx.x < nRanks) {
    peer = threadIdx.x;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr == nullptr)
      return;
    rocshmem::anvil::quiet(*hqPtr);
  }

  __syncthreads();

  if (threadIdx.x == 0) 
  	printf("WaitSignal RS thresh = %zu\n", val1);
  //for (int s = 0; s < nRanks; ++s) {
  if (threadIdx.x < nRanks) {
    s = threadIdx.x;
    if (s != myRank) {
    	uint64_t* waitAt = reinterpret_cast<uint64_t*>(localSignals) + s;
    	rocshmem::anvil::waitSignal(waitAt, kSignalRs);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
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
  }

  __syncthreads();
  //*val = *val + 1;
  anvilCrossRankBlockBarrier(nRanks, myRank, blockIdx.x, remoteBarriers, localBarriers, kBarrierVal0 + 1ull);

  //need a barrier
  // allgather
  //
  //for (int peer = 0; peer < nRanks; ++peer) {
  if (threadIdx.x < nRanks) {
    peer = threadIdx.x;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr == nullptr)
      return;
    rocshmem::anvil::SdmaQueueDeviceHandle& hq = *hqPtr;

    if (peer != myRank) {
    	void* peerBase = peerTempPtrs[peer];
    	char* peerC = reinterpret_cast<char*>(peerBase);
    	float* dst = reinterpret_cast<float*>(peerC) + (size_t)myRank * (size_t)chunk;
    	const float* src = recvbuff + (size_t)myRank * (size_t)chunk;

    	uint64_t* sigPeer = remoteSignals[peer];
    	uint64_t* sig = sigPeer + myRank;
    	rocshmem::anvil::putSignal(hq, dst, const_cast<float*>(src), chunkBytes, sig);
    } else {
	float* dst = tmpbuff + (size_t)myRank * (size_t)chunk;
        const float* src = recvbuff + (size_t)myRank * (size_t)chunk;
        rocshmem::anvil::putSelf(hq, dst, const_cast<float*>(src), chunkBytes);
    }
  }
  __syncthreads();

  if (threadIdx.x < nRanks) {
    peer = threadIdx.x;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr == nullptr)
      	return;
    rocshmem::anvil::quiet(*hqPtr);
  }
  __syncthreads();

  if (threadIdx.x == 0)
        printf("WaitSignal AG thresh = %zu\n", val1);
  if (threadIdx.x < nRanks) {
    s = threadIdx.x;
    if (s != myRank) {
    	uint64_t* waitAt = reinterpret_cast<uint64_t*>(localSignals) + s;
    	rocshmem::anvil::waitSignal(waitAt, kSignalAg);
    }
  }
  __syncthreads();

  if (threadIdx.x < nRanks) {
    s = threadIdx.x;
    if (s != myRank) {
    	const float* src =  tmpbuff + (size_t)s * (size_t)chunk;
    	float* dst = recvbuff + (size_t)s * (size_t)chunk;
    	for (int i = 0; i < chunk; ++i)
      		dst[i] = src[i];
    }
  }

  __syncthreads();
  anvilCrossRankBlockBarrier(nRanks, myRank, blockIdx.x, remoteBarriers, localBarriers, kBarrierVal0 + 2ull);

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
      comm->sdmaSyncBuffer == nullptr || comm->sdmaSyncBufferPeerPtrs_d == nullptr || 
      comm->sdmaBarrierBuffer == nullptr || comm->sdmaBarrierBufferPeerPtrs_d == nullptr || 
      comm->deviceHandles_d == nullptr)
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
  CUDACHECK(hipMemsetAsync(comm->sdmaBarrierBuffer, 0, comm->sdmaBarrierBufferBytes, stream));


  //printf("Invoking the kernel signal thresh = %zu\n", comm->sdmaAnvilSignalFlag);
  float* tmpBuf = reinterpret_cast<float*>(comm->sdmaFineGrainedTempBuf);
  hipLaunchKernelGGL(anvilTwoShotPhase1Kernel, dim3(1), dim3(8), 0, stream, sb, rb, tmpBuf, icount, nr,
                     comm->rank, comm->sdmaFineGrainedTempPeerPtrs_d, comm->deviceHandles_d, comm->sdmaSyncBufferPeerPtrs_d, 
		     comm->sdmaBarrierBufferPeerPtrs_d, comm->sdmaSyncBuffer, comm->sdmaBarrierBuffer, comm->sdmaAnvilBarrierFlag, comm->sdmaAnvilSignalFlag);
  //comm->sdmaAnvilBarrierFlag = comm->sdmaAnvilBarrierFlag + 3;
  //comm->sdmaAnvilSignalFlag = comm->sdmaAnvilSignalFlag + 2;

  CUDACHECK(hipGetLastError());


  return ncclSuccess;
}
