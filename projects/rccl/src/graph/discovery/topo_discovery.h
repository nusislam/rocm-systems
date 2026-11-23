/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_DISCOVERY_H_
#define NCCL_TOPO_DISCOVERY_H_

// Topology discovery: describes the node -- CPUs, PCI tree, GPUs and the devices of the network providers -- as
// topology XML, and fuses the XML of every rank sharing the node. It takes everything it needs from the
// communicator through ncclTopoDiscoveryConfig, so it makes no communicator, bootstrap or plugin-loader calls.

#include "../xml.h"
#include "nccl_net.h"
#include <map>
#include <mutex>

/***********************************************************/
/* Device discovery (topo_discovery_dev.cc)                */
/* CPUs, PCI tree and GPUs from sysfs, amd-smi/rocm-smi    */
/* and HIP; NICs placed in the PCI tree from their path.   */
/***********************************************************/

extern struct kvDict kvDictCpuArch[];

ncclResult_t ncclTopoRefreshBcmP2pLinks(void);
ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode);
ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* tagName, const char* pciPath, const char* netName,
                             struct ncclXmlNode** netNode, struct ncclXmlNode* forceParent = NULL);
ncclResult_t ncclTopoGetXmlCpuArch(ncclXml* xml, int* cpuArch);

/***********************************************************/
/* Network device import and NIC fusion                    */
/* (topo_discovery_net.cc)                                 */
/***********************************************************/

enum {
  NCCL_NET_MERGE_POLICY_ALL = 0,
  NCCL_NET_MERGE_POLICY_RAIL = 1
};

struct ncclTopoNetRailKeyList {
  std::map<uint64_t, int> keys;
};

// One network provider (net, gin, rma or collnet) to import into the topology.
struct ncclTopoNetInfo {
  bool coll;
  bool gin;
  bool rma;
  bool net;
  // communicator-specific information
  int netPluginIndex;
  int maxDevsPerNic;
  bool dmaBufSupport;
  // NIC fusion
  int mergeLevel;
  int mergePolicy;
  const char* forceMerge;
  // dev count tracking functions (not part of ncclNet)
  ncclResult_t (*getDevCount)(int, int*, int*);
  ncclResult_t (*setVirtDevCount)(int, int);
  // ncclNet API functions
  const char* name;
  ncclResult_t (*getProperties)(int, ncclNetProperties_t*);
  ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*);
  ncclResult_t (*devices)(int*);

  // system-wide list of unique rail keys
  struct ncclTopoNetRailKeyList* railKeyList;
};

extern struct kvDict nicPathKvList[];

ncclResult_t ncclTopoProcessNet(ncclXml* xml, const char* dumpXmlFile, struct ncclTopoNetInfo* net);
ncclResult_t ncclTopoForceMerge(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int* placedDevs,
                                ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs);
ncclResult_t ncclTopoGetFusionEnv(int* mergeLevel, const char** forceMerge, int cudaCompCap, int cpuArch);

/***********************************************************/
/* Fused XML construction (topo_discovery.cc)              */
/***********************************************************/

// Gathers perRankSize bytes from every rank of the fusion group into buf, which holds one slot per rank. The
// caller's own data is already in its slot.
typedef ncclResult_t (*ncclTopoXmlAllGather_t)(void* ctx, void* buf, size_t perRankSize);

struct ncclTopoDiscoveryConfig {
  // Topology XML to start from. NULL tries the default location.
  const char* xmlTopoFile;
  // Set when the XML is produced to be dumped to this file. Network devices are then kept physical.
  const char* dumpXmlFile;

  // The GPU this process drives. The other GPUs of the node are learnt through fusion.
  int64_t busId;
  int rank;
  int gdrSupport;
  int mloPart;
  // Lowest compute capability across ranks, which selects the NIC fusion defaults.
  int minCompCap;

  // Network providers, in precedence order. Their callbacks are serialized by netLock, which may be NULL.
  const struct ncclTopoNetInfo* nets;
  int nNets;
  std::mutex* netLock;

  // The ranks whose XML is fused, this one being localRank. allGather may be NULL for a single rank.
  int nLocalRanks;
  int localRank;
  ncclTopoXmlAllGather_t allGather;
  void* allGatherCtx;
  // Stamped on the fused <system> node.
  bool crossClique;
};

// Imports the devices of every provider in config->nets, filling in their NIC fusion settings and rail keys.
ncclResult_t ncclTopoImportNets(struct ncclXml* xml, const struct ncclTopoDiscoveryConfig* config, int cpuArch);

// Builds this process's XML: the topology file if any, its GPU and the network devices, trimmed to what is used.
ncclResult_t ncclTopoGetLocalXml(const struct ncclTopoDiscoveryConfig* config, struct ncclXml** xml);
// Replaces *xml, as returned by ncclTopoGetLocalXml, with the fusion of every local rank's XML. *xml stays owned
// by the caller, also on failure.
ncclResult_t ncclTopoFuseLocalXml(const struct ncclTopoDiscoveryConfig* config, struct ncclXml** xml);
// Both of the above. The caller frees *xml.
ncclResult_t ncclTopoDiscoverXml(const struct ncclTopoDiscoveryConfig* config, struct ncclXml** xml);

#endif
