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


RCCL_PARAM(AnvilAlltoAll, "ANVIL_ALLTOALL", 0);

// 16-byte vector element used for all global loads and stores.
using Vec16 = float4;

__device__ __forceinline__ Vec16 anvilLoadVec16(const float* ptr) {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    Vec16 f;
  } u;
  u.v = __builtin_amdgcn_global_load_b128((v4u_gptr)ptr, RCCL_SYSTEM_SYNCSCOPE);
  return u.f;
#else
  union {
    v4u v;
    Vec16 f;
  } u;
  u.v[0] = __builtin_nontemporal_load((u32_gptr)ptr + 0);
  u.v[1] = __builtin_nontemporal_load((u32_gptr)ptr + 1);
  u.v[2] = __builtin_nontemporal_load((u32_gptr)ptr + 2);
  u.v[3] = __builtin_nontemporal_load((u32_gptr)ptr + 3);
  return u.f;
#endif
}

__device__ __forceinline__ void anvilStoreVec16(float* ptr, Vec16 value) {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    Vec16 f;
  } u;
  u.f = value;
  __builtin_amdgcn_global_store_b128((v4u_gptr)ptr, u.v, RCCL_SYSTEM_SYNCSCOPE);
#else
  union {
    v4u v;
    Vec16 f;
  } u;
  u.f = value;
  __builtin_nontemporal_store(u.v[0], (u32_gptr)ptr + 0);
  __builtin_nontemporal_store(u.v[1], (u32_gptr)ptr + 1);
  __builtin_nontemporal_store(u.v[2], (u32_gptr)ptr + 2);
  __builtin_nontemporal_store(u.v[3], (u32_gptr)ptr + 3);
#endif
}

__device__ __forceinline__ Vec16 anvilAddVec16(Vec16 a, Vec16 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}


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
  if (numBlocks <= 1) {
    __syncthreads();
    return;
  }

  const int intraBase = kMaxSdmaBarrierBlocks * nRanks;

  //__syncthreads();
  /*if (threadIdx.x == 0) {
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
  }*/

  uint64_t* arrival = localBarriers + intraBase + 0;
  uint64_t* doneGen = localBarriers + intraBase + 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t prev = __hip_atomic_fetch_add(arrival, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (prev + 1 == (uint32_t)numBlocks) {
      __hip_atomic_store(arrival, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(doneGen, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      while (__hip_atomic_load(doneGen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != val) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
  }
  __syncthreads();
}


__global__ void anvilAlltoAllKernel(const float* __restrict__ sendbuff, float* __restrict__ recvbuff, 
					float* __restrict__ tmpbuff, int count, int nRanks, int myRank, 
					void** __restrict__ peerTempPtrs,
                                         rocshmem::anvil::SdmaQueueDeviceHandle** __restrict__ devHandles,
                                         uint64_t** __restrict__ remoteSignals, uint64_t** remoteBarriers, 
					 uint64_t* localSignals, uint64_t* localBarriers, uint64_t bar1, uint64_t bar2, uint64_t signal1, int warpSize, uint64_t bar3) {
  if (blockIdx.x >= 1)
    return;


  anvilCrossRankBlockBarrier(nRanks, myRank, blockIdx.x, remoteBarriers, localBarriers, bar1);

  const int chunk = count;
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

  if (laneId == 0 && warpId < nRanks) {
    peer = warpId;
    rocshmem::anvil::SdmaQueueDeviceHandle* hqPtr = devHandles[peer * kNumSdmaChannels + 0];
    if (hqPtr != nullptr)
       rocshmem::anvil::quiet(*hqPtr);
  }

  if (laneId == 0 && warpId < nRanks) {
    s = warpId;
    uint64_t* waitAt = reinterpret_cast<uint64_t*>(localSignals) + s;
    rocshmem::anvil::waitSignal(waitAt, signal1);
  }
  __syncthreads();
 }

  anvilIntraGpuBlockBarrier(nRanks, blockIdx.x, localBarriers, bar3);

  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
  __syncthreads();

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalThreads = blockDim.x * gridDim.x;
  const int vecCount = chunk / 4;

  const float* myTempFloat = reinterpret_cast<float*>(myTemp);
  for (int vi = gid; vi < vecCount; vi += totalThreads) {
    const int i = vi * 4;
#pragma unroll
    for (int r = 0; r < nRanks; ++r) {
      const float* srcRank = myTempFloat + (size_t)r * (size_t)chunk;
      anvilStoreVec16(recvbuff + i + r * chunk, anvilLoadVec16(srcRank + i));
    }
  }

  for (int i = vecCount * 4 + gid; i < chunk; i += totalThreads) {
#pragma unroll
    for (int r = 0; r < nRanks; ++r) {
      recvbuff[i + r * chunk] = myTempFloat[(size_t)r * (size_t)chunk + (size_t)i];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
  __syncthreads();

  //anvilIntraGpuBlockBarrier(nRanks, blockIdx.x, localBarriers, bar3+3);*/
  anvilCrossRankBlockBarrier(nRanks, myRank, blockIdx.x, remoteBarriers, localBarriers, bar2);

}

ncclResult_t rcclAnvilAlltoAllTry(const void* sendbuff, void* recvbuff, size_t count,
                                          ncclDataType_t datatype, ncclComm_t comm,
                                          hipStream_t stream, uint64_t bar1, uint64_t bar2, uint64_t signal1, uint64_t bar3) {
  if (rcclParamAnvilAlltoAll() == 0)
    return ncclInvalidUsage;
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr)
    return ncclInvalidArgument;
  if (datatype != ncclFloat)
    return ncclInvalidUsage;
  if (comm->nNodes != 1)
    return ncclInvalidUsage;

  const int nr = comm->nRanks;
  if (nr <= 0 || comm->sdmaFineGrainedTempBuf == nullptr || comm->sdmaFineGrainedTempPeerPtrs_d == nullptr ||
      comm->sdmaSyncBuffer == nullptr || comm->sdmaSyncBufferPeerPtrs_d == nullptr || 
      comm->sdmaBarrierBuffer == nullptr || comm->sdmaBarrierBufferPeerPtrs_d == nullptr || 
      comm->deviceHandles_d == nullptr)
    return ncclInvalidUsage;

  if (count > (size_t)INT_MAX)
    return ncclInvalidUsage;

  const int icount = (int)count;
  const size_t chunkBytes = (size_t)count * sizeof(float);
  const size_t stageBytes = (size_t)nr * chunkBytes;

  uint64_t **remoteBarriers = comm->remoteBarriers;
  uint64_t *localBarriers = comm->localBarriers;

  const float* sb = reinterpret_cast<const float*>(sendbuff);
  float* rb = reinterpret_cast<float*>(recvbuff);

  int warpSize = comm->WarpSize;
  float* tmpBuf = reinterpret_cast<float*>(comm->sdmaFineGrainedTempBuf);

  hipLaunchKernelGGL(anvilAlltoAllKernel, dim3(1), dim3(512), 0, stream, sb, rb, tmpBuf, icount, nr,
                     comm->rank, comm->sdmaFineGrainedTempPeerPtrs_d, comm->deviceHandles_d, comm->sdmaSyncBufferPeerPtrs_d, 
		     comm->sdmaBarrierBufferPeerPtrs_d, comm->sdmaSyncBuffer, comm->sdmaBarrierBuffer, bar1, bar2, signal1, warpSize, bar3);
  //hipMemcpyAsync(recvbuff, comm->sdmaFineGrainedTempBuf, count*comm->nRanks*sizeof(float), hipMemcpyDeviceToDevice, stream);

  CUDACHECK(hipGetLastError());


  return ncclSuccess;
}
