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
  //uint64_t kArrived = val;

  if (threadIdx.x < nRanks) {
      int peer = threadIdx.x;

      if (peer != myRank) {
      	uint64_t* p = remoteBarriers[peer] + slot;
      	__atomic_store_n(p, val, __ATOMIC_RELEASE);
      	
      	uint64_t* waitAt = localBarriers + blockId * nRanks + peer;
      	while (__atomic_load_n(waitAt, __ATOMIC_ACQUIRE) != val) {
	      __builtin_amdgcn_s_sleep(1);
      	}
      }
  }
  __syncthreads();
}

// Intra-GPU barrier across threadblocks launched on the same rank/device.
// Slots start at kMaxSdmaBarrierBlocks * nRanks + blockId (see init.cc barrier allocation).
__device__ __forceinline__ void anvilIntraGpuBlockBarrier(int nRanks, int blockId, uint64_t* localBarriers,
                                                            uint64_t val) {
  const int numBlocks = gridDim.x;
  if (numBlocks <= 1)
    return;

  const int intraBase = kMaxSdmaBarrierBlocks * nRanks;

  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t* arrival = localBarriers + intraBase + blockId;
    __atomic_store_n(arrival, val, __ATOMIC_RELEASE);

    for (int b = 0; b < numBlocks; ++b) {
      if (b == blockId)
        continue;
      uint64_t* waitAt = localBarriers + intraBase + b;
      while (__atomic_load_n(waitAt, __ATOMIC_ACQUIRE) != val) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
  }
  __syncthreads();
}


__global__ void anvilTwoShotPhase1Kernel(const float* __restrict__ sendbuff, float* __restrict__ recvbuff, 
					float* __restrict__ tmpbuff, int count, int nRanks, int myRank, 
					void** __restrict__ peerTempPtrs,
                                         rocshmem::anvil::SdmaQueueDeviceHandle** __restrict__ devHandles,
                                         uint64_t** __restrict__ remoteSignals, uint64_t** remoteBarriers, 
					 uint64_t* localSignals, uint64_t* localBarriers, uint64_t bar1, uint64_t bar2, uint64_t signal1, uint64_t signal2, uint64_t barMid, int warpSize, uint64_t bar3) {
  if (blockIdx.x >= 4)
    return;

  anvilCrossRankBlockBarrier(nRanks, myRank, blockIdx.x, remoteBarriers, localBarriers, bar1);

  const int chunk = count / nRanks;
  const size_t chunkBytes = (size_t)chunk * sizeof(float);
  char* myTemp = reinterpret_cast<char*>(peerTempPtrs[myRank]);
  int peer, s;

 if (blockIdx.x == 0) { 
  const int warpId = static_cast<int>(threadIdx.x) / warpSize;
  const int laneId = static_cast<int>(threadIdx.x) % warpSize; 

  if (laneId == 0 && warpId < nRanks) {
    peer = warpId;	 
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr != nullptr) {
    	rocshmem::anvil::SdmaQueueDeviceHandle& hq = *hqPtr;

    	void* peerBase = peerTempPtrs[peer];
    	char* peerC = reinterpret_cast<char*>(peerBase);
    	float* dst = reinterpret_cast<float*>(peerC) + (size_t)myRank * (size_t)chunk;
    	const float* src = sendbuff + (size_t)peer * (size_t)chunk;

    	uint64_t* sig = remoteSignals[peer] + myRank;
    	rocshmem::anvil::putSignal(hq, dst, const_cast<float*>(src), chunkBytes, sig);
    }
  }

  __syncthreads();

  if (laneId == 0 && warpId < nRanks) {
    peer = warpId;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr != nullptr)
       rocshmem::anvil::quiet(*hqPtr);
  }

  __syncthreads();

  if (laneId == 0 && warpId < nRanks) {
    s = warpId;
    uint64_t* waitAt = reinterpret_cast<uint64_t*>(localSignals) + s;
    rocshmem::anvil::waitSignal(waitAt, signal1);
  }
  __syncthreads();
 }

  anvilIntraGpuBlockBarrier(nRanks, blockIdx.x, localBarriers, bar3);

  //local reduction
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalThreads = blockDim.x * gridDim.x;

  for (int i = gid; i < chunk; i += totalThreads) {
    const size_t idx = (size_t)myRank * (size_t)chunk + (size_t)i;
    float acc = sendbuff[idx];
#pragma unroll
    for (int s = 0; s < nRanks; ++s) {
      if (s == myRank)
        continue;
      size_t idx1 = (size_t)s * (size_t)chunk + (size_t)i;
      acc += tmpbuff[idx1];
    }
    recvbuff[idx] = acc;
  }

  __syncthreads();
  anvilIntraGpuBlockBarrier(nRanks, blockIdx.x, localBarriers, bar3+1);
  anvilCrossRankBlockBarrier(nRanks, myRank, blockIdx.x, remoteBarriers, localBarriers, barMid);

  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }

  if (blockIdx.x == 0) {
  const int warpId = static_cast<int>(threadIdx.x) / warpSize;
  const int laneId = static_cast<int>(threadIdx.x) % warpSize;

  if (warpId < nRanks && laneId == 0) {
    peer = warpId;

    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr != nullptr) {
        rocshmem::anvil::SdmaQueueDeviceHandle& hq = *hqPtr;

    	void* peerBase = peerTempPtrs[peer];
    	//char* peerC = reinterpret_cast<char*>(peerBase) + (size_t)32*1024*1024;
    	char* peerC = reinterpret_cast<char*>(peerBase);

    	float* dst = reinterpret_cast<float*>(peerC) +  + (size_t)myRank * (size_t)chunk;

    	const float* src = recvbuff + (size_t)myRank * (size_t)chunk;

    	uint64_t* sigPeer = remoteSignals[peer];
    	uint64_t* sig = sigPeer + myRank;

    	rocshmem::anvil::putSignal(hq, dst, const_cast<float*>(src), chunkBytes, sig);
    }
  }
  __syncthreads();

  if (warpId < nRanks && laneId == 0) {
    peer = warpId;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr != nullptr)
    	rocshmem::anvil::quiet(*hqPtr);
  }
  __syncthreads();

  if (warpId < nRanks && laneId == 0) {
    	s = warpId;
    	uint64_t* waitAt = reinterpret_cast<uint64_t*>(localSignals) + s;

    	rocshmem::anvil::waitSignal(waitAt, (uint64_t) signal2);
  }
  __syncthreads();
  }

  anvilIntraGpuBlockBarrier(nRanks, blockIdx.x, localBarriers, bar3+2);

  for (int i = gid; i < chunk; i += totalThreads) {
#pragma unroll
    for (int r = 0; r < nRanks; ++r) {
      if (r == myRank)
  	continue;	      
      int srcRank = r;
      int destIdx = i + srcRank * chunk;
      int srcIdx;
      srcIdx = static_cast<int>(i);
      //char* tmpbuff = reinterpret_cast<char*>(myTemp) + (size_t)32*1024*1024;
      char* tmpbuff = reinterpret_cast<char*>(myTemp);

      float* src = reinterpret_cast<float*>(tmpbuff) + (size_t)r * (size_t)chunk;
      recvbuff[destIdx] = src[srcIdx];
    }
  }

  __syncthreads();


  anvilIntraGpuBlockBarrier(nRanks, blockIdx.x, localBarriers, bar3+3);
  anvilCrossRankBlockBarrier(nRanks, myRank, blockIdx.x, remoteBarriers, localBarriers, bar2);

}

ncclResult_t rcclAnvilTwoShotAllReduceTry(const void* sendbuff, void* recvbuff, size_t count,
                                          ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                          hipStream_t stream, uint64_t bar1, uint64_t bar2, uint64_t signal1, uint64_t signal2, uint64_t barMid, uint64_t bar3) {
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

  uint64_t **remoteBarriers = comm->remoteBarriers;
  uint64_t *localBarriers = comm->localBarriers;

  const float* sb = reinterpret_cast<const float*>(sendbuff);
  float* rb = reinterpret_cast<float*>(recvbuff);

  int warpSize = comm->WarpSize;
  float* tmpBuf = reinterpret_cast<float*>(comm->sdmaFineGrainedTempBuf);

  hipLaunchKernelGGL(anvilTwoShotPhase1Kernel, dim3(4), dim3(512), 0, stream, sb, rb, tmpBuf, icount, nr,
                     comm->rank, comm->sdmaFineGrainedTempPeerPtrs_d, comm->deviceHandles_d, comm->sdmaSyncBufferPeerPtrs_d, 
		     comm->sdmaBarrierBufferPeerPtrs_d, comm->sdmaSyncBuffer, comm->sdmaBarrierBuffer, bar1, bar2, signal1, signal2, barMid, warpSize, bar3);

  CUDACHECK(hipGetLastError());


  return ncclSuccess;
}
