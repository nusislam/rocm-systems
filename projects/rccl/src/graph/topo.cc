/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "cudawrap.h"
#include "graph.h"
#include "topo.h"
#include "discovery/topo_discovery.h"
#include <cstring>
#include "comm.h"
#include "nccl.h"
#include "nvmlwrap.h"
#include "coll_net.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin.h"
#endif
#include "rma.h"
#include "transport.h"
#include <sys/stat.h>
#include <fcntl.h>
#include "cpuset.h"
#include "bootstrap.h"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <float.h>

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char* topoNodeTypeStr[] = {"GPU", "PCI", "NVS", "CPU", "NIC", "NET", "GIN", "RMA", "DEV", "CXB"};
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
const char* topoLinkTypeStr[] = {"LOC", "XGMI", "", "C2C", "PCI", "", "", "", "", "SYS", "NET"};
const char* topoPathTypeStr[] = {"LOC", "XGMI", "NVB", "C2C", "PIX", "PXB", "P2C", "PXN", "PHB", "SYS", "NET", "DIS"};
#else
const char* topoLinkTypeStr[] = {"LOC", "NVL", "", "C2C", "PCI", "", "", "", "", "SYS", "NET"};
const char* topoPathTypeStr[] = {"LOC", "NVL", "NVB", "C2C", "PIX", "PXB", "P2C", "PXN", "PHB", "SYS", "NET", "DIS"};
#endif

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

static ncclResult_t findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu, struct ncclTopoNode* from) {
  *cpu = NULL;
  if (node->type == CPU) {
    *cpu = node;
    return ncclSuccess;
  }
  for (int l = 0; l < node->nlinks; l++) {
    // Go up the PCI tree to find the CPU. Follow only PCI switches.
    if (node->links[l].type == LINK_PCI && node->links[l].remNode != from &&
        (node->links[l].remNode->type == PCI || node->links[l].remNode->type == CPU)) {
      NCCLCHECK(findLocalCpu(node->links[l].remNode, cpu, node));
    }
    if (*cpu != NULL) return ncclSuccess;
  }
  return ncclSuccess;
}

int interCpuBw = 0;
int cpuPciBw = 0;

static ncclResult_t ncclTopoGetInterCpuBw(struct ncclTopoNode* cpu, float* bw) {
  *bw = LOC_BW;
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) {
    *bw = P9_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_ARM) {
    *bw = ARM_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    *bw = cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_ERP ? ERP_QPI_BW :
          cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SRP ? SRP_QPI_BW :
          cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SKL ? SKL_QPI_BW :
                                                            BDW_QPI_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
    // NCCL 2.31.2 modeled inter-socket rates. cpu.model is the Zen generation; Rome-class presets use cpu.rome.
    *bw = cpu->cpu.model == NCCL_TOPO_CPU_MODEL_AMD_ZEN5  ? AMD_ZEN5_BW :
          cpu->cpu.model == NCCL_TOPO_CPU_MODEL_AMD_ZEN34 ? AMD_ZEN34_BW :
                                                            AMD_ZEN12_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
    *bw = cpu->cpu.model == NCCL_TOPO_CPU_MODEL_YONGFENG ? YONGFENG_ZPI_BW : ZPI_BW;
  }
  return ncclSuccess;
}

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  for (int i = 0; i < system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *node = system->nodes[type].nodes + i;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
    WARN("Error : tried to create too many nodes of type %d", type);
    return ncclInternalError;
  }
  struct ncclTopoNode* n = system->nodes[type].nodes + system->nodes[type].count;
  system->nodes[type].count++;
  n->type = type;
  n->id = id;
  if (type == GPU) {
    n->gpu.dev = NCCL_TOPO_UNDEF;
    n->gpu.rank = NCCL_TOPO_UNDEF;
    n->gpu.cudaCompCap = NCCL_TOPO_UNDEF;
    n->gpu.mloPart = NCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    n->cpu.arch = NCCL_TOPO_UNDEF;
    n->cpu.vendor = NCCL_TOPO_UNDEF;
    n->cpu.model = NCCL_TOPO_UNDEF;
    n->cpu.rome = 0;
  } else if (type == NET) {
    n->net.vendor = UINT64_MAX;
    n->net.device = UINT64_MAX;
    n->net.asic = 0ULL;
    n->net.port = NCCL_TOPO_UNDEF;
    n->net.bw = 0.0;
    n->net.latency = 0.0;
    n->net.railId = NCCL_TOPO_UNDEF;
    n->net.planeId = NCCL_TOPO_UNDEF;
  } else if (type == GIN || type == RMA) {
    n->net.railId = NCCL_TOPO_UNDEF;
    n->net.planeId = NCCL_TOPO_UNDEF;
  } else if (type == DEV) {
    n->dev.dev = NCCL_TOPO_UNDEF;
    n->dev.cudaCompCap = NCCL_TOPO_UNDEF;
  }
  *node = n;
  return ncclSuccess;
}

ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
  struct ncclTopoNode* delNode = system->nodes[type].nodes + index;
  for (int t = 0; t < NCCL_TOPO_NODE_TYPES; t++) {
    if (delNode->paths[t] != nullptr) {
      WARN("Cannot remove topology node %d/%lx while paths are computed", type, delNode->id);
      return ncclInternalError;
    }
    for (int n = 0; n < system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes + n;
      if (node == delNode) continue;
      for (int l = 0; l < node->nlinks; l++) {
        while (l < node->nlinks && node->links[l].remNode == delNode) {
          memmove(node->links + l, node->links + l + 1, (node->nlinks - l - 1) * sizeof(struct ncclTopoLink));
          node->nlinks--;
        }
        if (l < node->nlinks && node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  if (type == DEV) {
    for (int n = 0; n < system->nodes[GPU].count; n++) {
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + n;
      if (gpu->gpu.parent == delNode) {
        gpu->gpu.parent = NULL;
      } else if (gpu->gpu.parent > delNode) {
        gpu->gpu.parent--;
      }
    }
  }
  memmove(delNode, delNode + 1, (system->nodes[type].count - index - 1) * sizeof(struct ncclTopoNode));
  system->nodes[type].count--;
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
  // Aggregate links into higher bw for NVLink
  struct ncclTopoLink* link;
  for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  if (link - node->links == NCCL_TOPO_MAX_LINKS) {
    WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
    return ncclInternalError;
  }
  if (link->remNode == NULL) node->nlinks++;
  link->type = type;
  link->remNode = remNode;
  link->bw += bw;

  // Sort links in BW descending order
  struct ncclTopoLink linkSave;
  memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
  while (link != node->links) {
    if ((link - 1)->bw >= linkSave.bw) break;
    memcpy(link, link - 1, sizeof(struct ncclTopoLink));
    link--;
  }
  memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
  return ncclSuccess;
}

// BCM Gen4 Switches present themselves as a two-level hierarchical switch
// even though they're supposed to sustain full BW across all ports.
// Flatten the switch as this extra level can break the search and make
// NCCL take wrong topology decisions.
int getBcmGen(uint64_t id, int level) {
  if ((id & 0xfffffffffffff000) == 0x1000c0101000a000) return 4;
  if ((id & 0xfffffffffffff000) == (0x1000c03010000000 | level * 0x1000)) return 5;
  return 0;
}
ncclResult_t ncclTopoFlattenBcmSwitches(struct ncclTopoSystem* system) {
  ncclResult_t ret = ncclSuccess;
  for (int s = 0; s < system->nodes[PCI].count; s++) {
    struct ncclTopoNode* pciSwitch = system->nodes[PCI].nodes + s;
    int gen = getBcmGen(pciSwitch->pci.device, 0);
    // Flatten Gen4 PEX switches in base mode
    if (gen) {
      // Find sub switches with the same device ID.
      int64_t* subSwIds;
      NCCLCHECK(ncclCalloc(&subSwIds, pciSwitch->nlinks));
      int subs = 0;
      for (int l = 0; l < pciSwitch->nlinks; l++) {
        struct ncclTopoNode* sub = pciSwitch->links[l].remNode;
        // Only fuse sub switches with the same device ID.
        if (sub->type != PCI || getBcmGen(sub->pci.device, 1) != gen) continue;
        // Save sub switch for later
        subSwIds[subs++] = sub->id;
        // Remove link to that sub switch
        memmove(pciSwitch->links + l, pciSwitch->links + l + 1,
                (pciSwitch->nlinks - l - 1) * (sizeof(struct ncclTopoLink)));
        pciSwitch->nlinks--;
        // Don't increase l for the next iteration as we just shifted all links by one.
        l--;
      }

      for (int s = 0; s < subs; s++) {
        // Find sub switch (system->nodes[PCI].nodes is changing every time we remove a node)
        int index;
        NCCLCHECKGOTO(ncclTopoIdToIndex(system, PCI, subSwIds[s], &index), ret, fail);
        struct ncclTopoNode* sub = system->nodes[PCI].nodes + index;
        // Connect all sub PCI devices to the parent switch
        for (int l = 0; l < sub->nlinks; l++) {
          struct ncclTopoNode* remNode = sub->links[l].remNode;
          if (remNode == pciSwitch) continue;
          // Add link from parent PCI switch -> PCI device
          if (pciSwitch->nlinks == NCCL_TOPO_MAX_LINKS) {
            WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
            ret = ncclInternalError;
            goto fail;
          }
          memcpy(pciSwitch->links + pciSwitch->nlinks, sub->links + l, sizeof(struct ncclTopoLink));
          pciSwitch->nlinks++;
          // Update link from PCI device -> parent PCI switch
          for (int rl = 0; rl < remNode->nlinks; rl++) {
            if (remNode->links[rl].remNode == sub) {
              remNode->links[rl].remNode = pciSwitch;
              break;
            }
          }
        }
        NCCLCHECKGOTO(ncclTopoRemoveNode(system, PCI, index), ret, fail);
      }
      // Set subdevice to 0xffff to make sure we don't merge this switch again.
      pciSwitch->pci.device |= 0xffff;
      free(subSwIds);
      // Restart, as system->nodes[PCI].nodes has changed.
      s = -1;  // Will be incremented to 0 in the next loop iteration
      continue;
    fail:
      free(subSwIds);
      return ret;
    }
  }
  return ret;
}

ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
  // And connect all CPU nodes together
  for (int n = 0; n < system->nodes[CPU].count; n++) {
    struct ncclTopoNode* cpu1 = system->nodes[CPU].nodes + n;
    for (int p = 0; p < system->nodes[CPU].count; p++) {
      struct ncclTopoNode* cpu2 = system->nodes[CPU].nodes + p;
      if (n == p || (NCCL_TOPO_ID_SYSTEM_ID(cpu1->id) != NCCL_TOPO_ID_SYSTEM_ID(cpu2->id))) continue;
      float bw;
      NCCLCHECK(ncclTopoGetInterCpuBw(cpu1, &bw));
      NCCLCHECK(ncclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    char mloStr[128] = "";
    snprintf(mloStr, sizeof(mloStr), " [MLOPart %d]", node->gpu.mloPart);
    sprintf(line + offset, "%s/%lx-%lx (%d)%s", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->gpu.rank, (node->gpu.mloPart == -1) ? "" : mloStr);
  } else if (node->type == CPU) {
    sprintf(line + offset, "%s/%lx-%lx (%d/%d/%d)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->cpu.arch, node->cpu.vendor, node->cpu.model);
  } else if (node->type == PCI) {
    sprintf(line + offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->pci.device);
  } else if (node->type == DEV) {
    sprintf(line + offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->dev.device);
  } else {
    sprintf(line + offset, "%s/%lx-%lx", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id));
  }
  INFO(NCCL_GRAPH, "%s", line);
  for (int i = 0; i < offset; i++) line[i] = ' ';

  for (int l = 0; l < node->nlinks; l++) {
    struct ncclTopoLink* link = node->links + l;
    if (link->type == LINK_LOC) {
      sprintf(line + offset, "+ %s[%2.1f] - %s/%lx-%lx", topoLinkTypeStr[link->type], link->bw,
              topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id),
              NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
      INFO(NCCL_GRAPH, "%s", line);
    } else if (link->type != LINK_PCI || link->remNode != prevNode) {
      sprintf(line + offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type], link->bw);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line + nextOffset, "%s/%lx-%lx (%d/%lx/%d/%f)", topoNodeTypeStr[link->remNode->type],
                  NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id),
                  link->remNode->net.collSupport, link->remNode->net.asic, link->remNode->net.port,
                  link->remNode->net.bw);
        } else {
          sprintf(line + nextOffset, "%s/%lx-%lx", topoNodeTypeStr[link->remNode->type],
                  NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
        }
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxBw %2.1f totalBw %2.1f ===", s->maxBw, s->totalBw);
  char line[2048];
  for (int n = 0; n < s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes + n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  NCCLCHECK(ncclTopoPrintPaths(s));
  return ncclSuccess;
}

static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
  // Shift all links to have upLink as last link
  if (upNode) {
    int l = 0;
    while (node->links[l].remNode != upNode) l++;
    struct ncclTopoLink upLink;
    memcpy(&upLink, node->links + l, sizeof(struct ncclTopoLink));
    while (node->links[l + 1].remNode) {
      memcpy(node->links + l, node->links + l + 1, sizeof(struct ncclTopoLink));
      l++;
    }
    memcpy(node->links + l, &upLink, sizeof(struct ncclTopoLink));
  }

  // Recursively sort the PCI tree
  for (int l = 0; l < node->nlinks; l++) {
    struct ncclTopoLink* link = node->links + l;
    if (link->type == LINK_PCI && link->remNode != upNode) NCCLCHECK(ncclTopoSort(link->remNode, node));
  }
  return ncclSuccess;
}

// We want the graph to be organized to ease/accelerate traversal :
// 1. NVLinks (already the case)
// 2. PCI down
// 3. PCI up
// 4. SYS (already the case)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n = 0; n < system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes + n, NULL));
  return ncclSuccess;
}

// Minimum network BW of a single device accessible by rank.
// Note: This function does not sum up the bw over multiple NICs if many are accessible.
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw) {
  int g = 0;
  while (g < system->nodes[GPU].count && system->nodes[GPU].nodes[g].gpu.rank != rank) g++;
  if (g == system->nodes[GPU].count) return ncclInternalError;

  // GPUs with no reachable NET node have a minimum net bw of 0.
  int localNets[NCCL_TOPO_MAX_NODES];
  int localNetCount = 0;
  NCCLCHECK(ncclTopoGetLocal(system, GPU, g, NET, localNets, &localNetCount, NULL));
  if (localNetCount == 0) {
    *bw = 0.0;
    return ncclSuccess;
  }

  int64_t firstNetId = 0;
  float minBw = FLT_MAX;
  for (int c = 0; c < MAXCHANNELS; c++) {
    int net;
    int64_t netId;
    NCCLCHECK(ncclTopoGetLocalNet(system, rank, c, &netId, NULL));
    NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &net));
    if (c == 0) firstNetId = netId;
    else if (netId == firstNetId) break;

    minBw = std::min(minBw, system->nodes[GPU].nodes[g].paths[NET][net].bw);
  }
  // if no net is found, return 0 as a minimum bw
  *bw = (minBw < FLT_MAX) ? minBw : 0.0;
  return ncclSuccess;
}

static ncclResult_t ncclTopoAddNetAsic(struct ncclXmlNode* xmlNet, int64_t netId, struct ncclTopoNode* net) {
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlNet, "guid", &str));
  // if not guid is present use the net->id unique id instead, which will be unique within the node/NVLD
  net->net.asic = (str) ? strtoull(str, NULL, 16) : netId;
  return ncclSuccess;
}

static ncclResult_t ncclTopoAddNet(struct ncclXmlNode* xmlNet, struct ncclXmlNode* parent,
                                   struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId,
                                   int64_t busId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, NET, netId));
  net->net.dev = dev;

  int mbps;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0), NCCL_GRAPH);
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;
  ncclResult_t ret;
  NOWARN(ret = xmlGetAttrFloat(xmlNet, "latency", &net->net.latency), NCCL_GRAPH);
  if (ret != ncclSuccess) net->net.latency = 0;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "port", &net->net.port, 0), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "gdr", &net->net.gdrSupport, 0), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "maxconn", &net->net.maxChannels, MAXCHANNELS), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "coll", &net->net.collSupport, 0), NCCL_GRAPH);
  net->net.busId = busId;  // RCCL: keep this

  NCCLCHECK(ncclTopoAddNetAsic(xmlNet, netId, net));
  NCCLCHECK(xmlGetAttrInt(xmlNet, "rail", &net->net.railId));
  NCCLCHECK(xmlGetAttrInt(xmlNet, "plane", &net->net.planeId));

  // build the PCI id using the parent PCI link
  const char* pciBusId = NULL;
  if (parent) {
    NCCLCHECK(xmlGetAttr(parent, "busid", &pciBusId));
    NCCLCHECK(xmlGetAttrUint64Default(parent, "vendor", &net->net.vendor, UINT64_MAX));
    NCCLCHECK(xmlGetAttrUint64Default(parent, "device", &net->net.device, UINT64_MAX));
  }
  // If we fail to find the PCIe path, we use the GUID instead.
  uint64_t hacc[2] = {1, 1};
  if (pciBusId) eatHash(hacc, pciBusId, strlen(pciBusId));
  else eatHash(hacc, &net->net.asic);
  net->net.pciId = digestHash(hacc);

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

// Create a GIN topology node for a gin-capable NIC. NCCL 2.29 decouples GIN
// from the NET plugin: GIN devices live in system->nodes[GIN] (parallel to
// system->nodes[NET]) and are looked up by ncclTopoGetLocalGinDev for
// device-API kernels (-D 3 / -D 4 in rccl-tests).
static ncclResult_t ncclTopoAddGin(struct ncclXmlNode* xmlNet, struct ncclXmlNode* parent,
                                   struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, GIN, netId));
  net->net.dev = dev;

  int mbps;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0), NCCL_GRAPH);
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;

  if (parent) {
    NCCLCHECK(xmlGetAttrUint64Default(parent, "vendor", &net->net.vendor, UINT64_MAX));
    NCCLCHECK(xmlGetAttrUint64Default(parent, "device", &net->net.device, UINT64_MAX));
  }

  NCCLCHECK(ncclTopoAddNetAsic(xmlNet, netId, net));
  NCCLCHECK(xmlGetAttrInt(xmlNet, "rail", &net->net.railId));
  NCCLCHECK(xmlGetAttrInt(xmlNet, "plane", &net->net.planeId));

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

static ncclResult_t ncclTopoAddRma(struct ncclXmlNode* xmlNet, struct ncclXmlNode* parent,
                                   struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, RMA, netId));
  net->net.dev = dev;

  int mbps;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0), NCCL_GRAPH);
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;

  if (parent) {
    NCCLCHECK(xmlGetAttrUint64Default(parent, "vendor", &net->net.vendor, UINT64_MAX));
    NCCLCHECK(xmlGetAttrUint64Default(parent, "device", &net->net.device, UINT64_MAX));
  }

  NCCLCHECK(ncclTopoAddNetAsic(xmlNet, netId, net));
  NCCLCHECK(xmlGetAttrInt(xmlNet, "rail", &net->net.railId));
  NCCLCHECK(xmlGetAttrInt(xmlNet, "plane", &net->net.planeId));

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNic(struct ncclXmlNode* xmlNic, struct ncclTopoSystem* system, struct ncclTopoNode* nic,
                            int systemId, int64_t busId) {
  // Find the closest PCI ancestor to the NIC
  struct ncclXmlNode* parent = xmlNic->parent;
  while (parent != NULL && strcmp(parent->name, "pci") != 0) parent = parent->parent;

  for (int s = 0; s < xmlNic->nSubs; s++) {
    struct ncclXmlNode* xmlNet = xmlNic->subs[s];
    int net = strcmp(xmlNet->name, "net") == 0;
    int gin = strcmp(xmlNet->name, "gin") == 0;
    int rma = strcmp(xmlNet->name, "rma") == 0;
    if (!net && !gin && !rma) continue;

    // Physical NICs fused into a vNIC are marked net="0"; absent "net" remains enabled for older XML.
    if (net) NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "net", &net, 1));

    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlNet, "dev", &index));
    // XML net nodes without "dev" attribute will be ignore
    if (index == -1) continue;
    if (net) NCCLCHECK(ncclTopoAddNet(xmlNet, parent, system, nic, systemId, busId));
    if (gin) NCCLCHECK(ncclTopoAddGin(xmlNet, parent, system, nic, systemId));
    if (rma) NCCLCHECK(ncclTopoAddRma(xmlNet, parent, system, nic, systemId));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // There is no direct mapping between CUDA SM to HIP GFX. Use SM60 as compatibility level.
  gpu->gpu.cudaCompCap = 60;
  // Repurpose previously unused "sm" as CU counts
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.cu));
#else
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.cudaCompCap));
#endif
  const char* gcnArch;
  const char* gcnArchName;
  NCCLCHECK(xmlGetAttrStr(xmlGpu, "gcn", &gcnArch));
  convertGcnArchToGcnArchName(gcnArch, &gcnArchName);
  strncpy(gpu->gpu.gcn, gcnArchName, GCN_ARCH_NAME_LEN - 1);
  gpu->gpu.gcn[GCN_ARCH_NAME_LEN - 1] = '\0';
  rcclHipDeviceArch_t arch;
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "arch", &arch.value));
  memcpy(&gpu->gpu.arch, &arch.arch, sizeof(hipDeviceArch_t));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "rank", &gpu->gpu.rank));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpu->gpu.dev));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpu->gpu.gdrSupport));
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &gpu->gpu.mloPart, NCCL_TOPO_UNDEF));
  // Do not go any further, nvlinks will be added in a second pass
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetIntDevice(struct ncclXmlNode* xmlPci, uint64_t* device) {
  uint64_t ret = 0;
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
  if (str) ret += strtol(str, NULL, 0) << 48;
  NCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
  if (str) ret += strtol(str, NULL, 0) << 32;
  NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
  if (str) ret += strtol(str, NULL, 0) << 16;
  NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
  if (str) ret += strtol(str, NULL, 0);
  *device = ret;
  return ncclSuccess;
}

struct kvDict kvDictPciClass[] = {{PCI_BRIDGE_DEVICE_CLASS, PCI},
                                  {"0x080100", /*CX8 data direct*/ PCI},
                                  {"0x068000", NVS},
                                  {"0x068001", CPU},
                                  {"0x03", GPU},
                                  {"0x02", NIC},
                                  {"0x120000", GPU},
                                  {NULL, PCI /* Default fallback value */}};
struct kvDict kvDictPciGen[] = {{"2.5 GT/s", 15},
                                {"5 GT/s", 30},
                                {"8 GT/s", 60},
                                {"16 GT/s", 120},
                                {"32 GT/s", 240}, /* Kernel 5.6 and earlier */
                                {"2.5 GT/s PCIe", 15},
                                {"5.0 GT/s PCIe", 30},
                                {"8.0 GT/s PCIe", 60},
                                {"16.0 GT/s PCIe", 120},
                                {"32.0 GT/s PCIe", 240},
                                {"64.0 GT/s PCIe", 480},
                                {NULL, 60 /* Default fallback */}}; // x100 Mbps per lane

// Return all DEV nodes whose id matches baseId with mlopart bits masked out.
ncclResult_t ncclTopoGetDevNodes(struct ncclTopoSystem* system, int64_t baseId, struct ncclTopoNode** nodes,
                                 int* nNodes) {
  *nNodes = 0;
  int64_t maskedBase = baseId & ~(int64_t)NCCL_TOPO_MLOPART_MASK;
  for (int d = 0; d < system->nodes[DEV].count; d++) {
    struct ncclTopoNode* dev = &system->nodes[DEV].nodes[d];
    if ((dev->id & ~(int64_t)NCCL_TOPO_MLOPART_MASK) == maskedBase) {
      if (*nNodes == NCCL_TOPO_MLOPART_DEV_MAX) {
        WARN("ncclTopoGetDevNodes : too many DEV nodes for busId %lx (max %d)", baseId, NCCL_TOPO_MLOPART_DEV_MAX);
        return ncclInternalError;
      }
      if (nodes) nodes[(*nNodes)] = dev;
      (*nNodes)++;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoCheckMloPartBusId(int64_t busId) {
  // check that the bits used for mlopart information are free and always 0 to avoid collision.
  if (busId & NCCL_TOPO_MLOPART_MASK) {
    WARN(
      "BusId 0x%lx has non-zero bits in MLOPart mask 0x%llx, cannot encode MLOPart partition index without collision",
      busId, (long long)NCCL_TOPO_MLOPART_MASK);
    return ncclInternalError;
  }
  return ncclSuccess;
}

NCCL_PARAM(TopoSplitMlopart, "TOPO_SPLIT_MLOPART", 1);

static ncclResult_t ncclTopoAddGpuSub(struct ncclXmlNode* xmlPci, struct ncclXmlNode* xmlGpu,
                                      struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId,
                                      int64_t busId, float bw) {
  int mloPart = 0, sm = 0;
  int64_t devBusId = busId;
  NCCLCHECK(ncclTopoCheckMloPartBusId(busId));
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &mloPart, NCCL_TOPO_UNDEF));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &sm));
  if (mloPart != NCCL_TOPO_UNDEF && ncclParamTopoSplitMlopart()) {
    if (mloPart >= NCCL_TOPO_MLOPART_DEV_MAX) {
      WARN("MLOPart index %d out of range (max %d)", mloPart, NCCL_TOPO_MLOPART_DEV_MAX - 1);
      return ncclInternalError;
    }
    devBusId = NCCL_TOPO_MLOPART_BUSID(busId, mloPart);
  }

  struct ncclTopoNode* gpudeviceNode = NULL;
  NCCLCHECK(ncclTopoGetNode(system, &gpudeviceNode, DEV, NCCL_TOPO_ID(systemId, devBusId)));
  if (gpudeviceNode == NULL) {
    NCCLCHECK(ncclTopoCreateNode(system, &gpudeviceNode, DEV, NCCL_TOPO_ID(systemId, devBusId)));
    NCCLCHECK(ncclTopoGetIntDevice(xmlPci, &gpudeviceNode->dev.device));
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // "sm" is repurposed as CU count on AMD, so pin the DEV node to SM60 like
    // ncclTopoAddGpu instead of storing the CU count as a compute capability.
    gpudeviceNode->dev.cudaCompCap = 60;
#else
    NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpudeviceNode->dev.cudaCompCap));
#endif
    NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpudeviceNode->dev.dev));
    NCCLCHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpudeviceNode->dev.gdrSupport));
    NCCLCHECK(ncclTopoConnectNodes(gpudeviceNode, parent, LINK_PCI, bw));
    NCCLCHECK(ncclTopoConnectNodes(parent, gpudeviceNode, LINK_PCI, bw));
    // Sibling partitions of this device are linked once the XGMI pass has run, see
    // ncclTopoConnectMloPartSiblings().
  }

  struct ncclTopoNode* gpuNode = NULL;
  NCCLCHECK(ncclTopoCreateNode(system, &gpuNode, GPU,
                               NCCL_TOPO_ID(systemId, NCCL_TOPO_GPU_LOCAL_ID(devBusId, gpudeviceNode->dev.nGpus))));
  NCCLCHECK(ncclTopoAddGpu(xmlGpu, system, gpuNode));
  gpuNode->gpu.parent = gpudeviceNode;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // RCCL: mirror the GPU rank's GCN arch onto the parent DEV node. XGMI link BW is keyed on
  // the GCN arch string (not cudaCompCap), and v2.30's DEV refactor attaches inter-device
  // links to DEV nodes, so ncclTopoAddXGMI needs the arch available on the DEV node.
  strncpy(gpudeviceNode->dev.gcn, gpuNode->gpu.gcn, GCN_ARCH_NAME_LEN);
#endif
  NCCLCHECK(ncclTopoConnectNodes(gpudeviceNode, gpuNode, LINK_LOC, LOC_BW));
  NCCLCHECK(ncclTopoConnectNodes(gpuNode, gpudeviceNode, LINK_LOC, LOC_BW));
  gpudeviceNode->dev.nGpus++;
  return ncclSuccess;
}

ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent,
                            int systemId, int numaId) {
  const char* str;

  int type;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "class", &str));
  NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));

  int64_t busId;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
  NCCLCHECK(busIdToInt64(str, &busId));

  float bw;
  {
    int width, speed;
    NCCLCHECK(xmlGetAttrInt(xmlPci, "link_width", &width));
    NCCLCHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));
    // Manage cases where speed was not indicated in /sys
    if (width == 0) width = 16;
    NCCLCHECK(kvConvertToInt(str, &speed, kvDictPciGen)); // Values in 100Mbps, per lane (we want GB/s in the end)
    bw = width * speed / 80.0;
  }

  for (int g = 0; g < xmlPci->nSubs; ++g) {
    if (strcmp(xmlPci->subs[g]->name, "gpu") != 0) continue;
    struct ncclXmlNode* xmlGpu = xmlPci->subs[g];
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlGpu, "rank", &index));
    if (index == -1) return ncclSuccess;
    NCCLCHECK(ncclTopoAddGpuSub(xmlPci, xmlGpu, system, parent, systemId, busId, bw));
  }

  struct ncclTopoNode* node = NULL;
  struct ncclXmlNode* xmlNic = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
  if (xmlNic != NULL) {
    type = NIC;
    // Ignore sub device ID and merge multi-port NICs into one PCI device.
    struct ncclTopoNode* nicNode = NULL;
    int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, busId);
    int64_t id = NCCL_TOPO_ID(systemId, localNicId);
    NCCLCHECK(ncclTopoGetNode(system, &nicNode, type, id));
    if (nicNode == NULL) {
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, type, id));
      node = nicNode; // Connect it to parent later on
    }
    NCCLCHECK(ncclTopoAddNic(xmlNic, system, nicNode, systemId, busId));
  } else if (type == PCI) {
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
    NCCLCHECK(ncclTopoGetIntDevice(xmlPci, &node->pci.device));

    for (int s = 0; s < xmlPci->nSubs; s++) {
      struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
      if (strcmp(xmlSubPci->name, "pcilink") != 0) {
        // PCI links will be added later
        NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId, numaId));
      }
    }
  }

  if (node) {
    NCCLCHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, bw));
    NCCLCHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, bw));
  }
  return ncclSuccess;
}

struct kvDict kvDictCpuVendor[] = {{"GenuineIntel", NCCL_TOPO_CPU_VENDOR_INTEL},
                                   {"AuthenticAMD", NCCL_TOPO_CPU_VENDOR_AMD},
                                   {"CentaurHauls", NCCL_TOPO_CPU_VENDOR_ZHAOXIN},
                                   {"  Shanghai  ", NCCL_TOPO_CPU_VENDOR_ZHAOXIN},
                                   {NULL, 0}};

// RCCL: Rome class is family 17h model >= 0x31, or all of 19h; also accepts the old file encoding (0x8F/0xAF).
static bool rcclAmdCpuIsRome(int familyId, int modelId) {
  if (familyId == 0x17 || familyId == 0x8F) return modelId >= 0x31;
  return familyId == 0x19 || familyId == 0xAF;
}

ncclResult_t ncclGetSystemId(struct ncclTopoSystem* system, struct ncclXmlNode* xmlCpu, int* systemIdPtr) {
  const char* hostHashStr;
  NCCLCHECK(xmlGetAttr(xmlCpu, "host_hash", &hostHashStr));
  uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, NULL, 16) : 0;
  int systemId;
  for (systemId = 0; systemId < system->nHosts; systemId++)
    if (system->hostHashes[systemId] == hostHash) break;
  if (systemId == system->nHosts) system->hostHashes[system->nHosts++] = hostHash;
  *systemIdPtr = systemId;
  return ncclSuccess;
}

ncclResult_t ncclTopoAddCpu(struct ncclXmlNode* xmlCpu, struct ncclTopoSystem* system) {
  int numaId;
  NCCLCHECK(xmlGetAttrInt(xmlCpu, "numaid", &numaId));
  int systemId;
  NCCLCHECK(ncclGetSystemId(system, xmlCpu, &systemId));
  struct ncclTopoNode* cpu;
  NCCLCHECK(ncclTopoCreateNode(system, &cpu, CPU, NCCL_TOPO_ID(systemId, numaId)));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlCpu, "affinity", &str));
  if (str != NULL) {
    int cpuOffset;
    NCCLCHECK(xmlGetAttrIntDefault(xmlCpu, "affinity_offset", &cpuOffset, 0));
    NCCLCHECK(ncclStrToCpuset(str, &cpu->cpu.affinity, cpuOffset));
  }

  NCCLCHECK(xmlGetAttrStr(xmlCpu, "arch", &str));
  NCCLCHECK(kvConvertToInt(str, &cpu->cpu.arch, kvDictCpuArch));
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86) {
    NCCLCHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));
    NCCLCHECK(kvConvertToInt(str, &cpu->cpu.vendor, kvDictCpuVendor));
    if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      // Granite Rapids (0xAD/0xAE) and Sierra Forest (0xAF) are newer than Emerald Rapids but carry LOWER model IDs
      cpu->cpu.model = (familyId == 6 && (modelId >= 0xCF || modelId == 0xAD || modelId == 0xAE || modelId == 0xAF)) ?
                         NCCL_TOPO_CPU_MODEL_INTEL_ERP :
                       (familyId == 6 && modelId >= 0x8F) ? NCCL_TOPO_CPU_MODEL_INTEL_SRP :
                       (familyId == 6 && modelId >= 0x55) ? NCCL_TOPO_CPU_MODEL_INTEL_SKL :
                                                            NCCL_TOPO_CPU_MODEL_INTEL_BDW;
    } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
      int familyId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      // AMD Zen 1/2 (Naples/Rome):  CPUID Family 17h = 23
      // AMD Zen 3/4 (Milan/Genoa):  CPUID Family 19h = 25
      // AMD Zen 5   (Turin):        CPUID Family 1Ah = 26
      cpu->cpu.model =
        // Old calculations to support injected topology
        (familyId == 0xBF) ? NCCL_TOPO_CPU_MODEL_AMD_ZEN5 :
        (familyId == 0xAF) ? NCCL_TOPO_CPU_MODEL_AMD_ZEN34 :
        (familyId == 0x8F) ? NCCL_TOPO_CPU_MODEL_AMD_ZEN12 :
                             // CPUID Family
          (familyId == 26) ? NCCL_TOPO_CPU_MODEL_AMD_ZEN5 :
        (familyId == 25)   ? NCCL_TOPO_CPU_MODEL_AMD_ZEN34 :
                             NCCL_TOPO_CPU_MODEL_AMD_ZEN12;
    } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      if (familyId == 7 && modelId == 0x5B) cpu->cpu.model = NCCL_TOPO_CPU_MODEL_YONGFENG;
    }
    if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      cpu->cpu.rome = rcclAmdCpuIsRome(familyId, modelId);
    }
  }
  int nicIndex = 0;
  for (int s = 0; s < xmlCpu->nSubs; s++) {
    struct ncclXmlNode* node = xmlCpu->subs[s];
    if (strcmp(node->name, "pci") == 0) NCCLCHECK(ncclTopoAddPci(node, system, cpu, systemId, numaId));
    if (strcmp(node->name, "nic") == 0) {
      struct ncclTopoNode* nic = NULL;
      // NICs under <cpu> have no busid, index them to keep several on one NUMA node distinct
      int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, nicIndex++);
      int64_t id = NCCL_TOPO_ID(systemId, localNicId);
      NCCLCHECK(ncclTopoGetNode(system, &nic, NIC, id));
      if (nic == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &nic, NIC, id));
        NCCLCHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_BW));
        NCCLCHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_BW));
      }
      NCCLCHECK(ncclTopoAddNic(node, system, nic, systemId, 0));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetGpuDevNode(struct ncclXmlNode* xmlGpu, const char* busId, int systemId,
                                          struct ncclTopoSystem* system, struct ncclTopoNode** devNode) {
  int mloPart;
  int64_t rawBusId;
  NCCLCHECK(busIdToInt64(busId, &rawBusId));
  NCCLCHECK(ncclTopoCheckMloPartBusId(rawBusId));
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &mloPart, NCCL_TOPO_UNDEF));
  int64_t devBusId =
    (mloPart != NCCL_TOPO_UNDEF && ncclParamTopoSplitMlopart()) ? NCCL_TOPO_MLOPART_BUSID(rawBusId, mloPart) : rawBusId;
  NCCLCHECK(ncclTopoGetNode(system, devNode, DEV, NCCL_TOPO_ID(systemId, devBusId)));
  return ncclSuccess;
}

// Return true only for the first GPU sibling (in XML order) that shares the same DEV node.
static ncclResult_t ncclTopoXmlIsPrimaryGpuForDev(struct ncclXmlNode* xmlGpu, bool* isPrimary) {
  *isPrimary = true;
  int myMloPart = NCCL_TOPO_UNDEF;
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &myMloPart, NCCL_TOPO_UNDEF));
  for (int s = 0; s < xmlGpu->parent->nSubs; s++) {
    struct ncclXmlNode* sib = xmlGpu->parent->subs[s];
    if (strcmp(sib->name, "gpu") != 0) continue;
    if (sib == xmlGpu) break;
    int sibMloPart = NCCL_TOPO_UNDEF;
    NCCLCHECK(xmlGetAttrIntDefault(sib, "mlopart", &sibMloPart, NCCL_TOPO_UNDEF));
    if (sibMloPart == myMloPart || !ncclParamTopoSplitMlopart()) {
      *isPrimary = false;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
static ncclResult_t ncclTopoGetDevForPciBusId(struct ncclTopoSystem* system, int systemId, int64_t busId,
                                              struct ncclTopoNode** devNode) {
  *devNode = NULL;
  NCCLCHECK(ncclTopoGetNode(system, devNode, DEV, NCCL_TOPO_ID(systemId, busId)));
  if (*devNode) return ncclSuccess;
  // HIP uses the function nibble as the partition index. Match an overlaid DEV
  // on the same PCI domain:bus:device, regardless of its physical function.
  int fn = (int)(busId & 0xf);
  if (fn >= NCCL_TOPO_MLOPART_DEV_MAX) return ncclSuccess;
  int64_t phys = busId & ~0xfLL;
  for (int d = 0; d < system->nodes[DEV].count; d++) {
    struct ncclTopoNode* candidate = system->nodes[DEV].nodes + d;
    int64_t local = NCCL_TOPO_ID_LOCAL_ID(candidate->id);
    if (NCCL_TOPO_ID_SYSTEM_ID(candidate->id) == systemId &&
        (local & ~(NCCL_TOPO_MLOPART_MASK | 0xfLL)) == phys &&
        (local & NCCL_TOPO_MLOPART_MASK) == NCCL_TOPO_MLOPART(fn)) {
      *devNode = candidate;
      break;
    }
  }
  return ncclSuccess;
}

// RCCL: workaround (AICOMNET-435) until fused MNNVL hosts are recognised as NVL-connected. Only a
// scale-out comm (cross_clique on <system>, set by ncclTopoDiscoverXml) links UALoE fabric peers through
// the shared NVS; a comm inside one clique keeps the per-host topology its ring search and channel
// setup are tuned for.
static ncclResult_t rcclTopoXmlIsCrossClique(struct ncclXmlNode* node, bool* crossClique) {
  struct ncclXmlNode* topNode = node;
  while (topNode->parent) topNode = topNode->parent;
  int value = 0;
  NCCLCHECK(xmlGetAttrIntDefault(topNode, "cross_clique", &value, 0));
  *crossClique = value == 1;
  return ncclSuccess;
}

// RCCL: sum of the link counts of the fabric peer entries of the GPU that owns an xgmi node, which is what the
// default (count * XGMI speed) DEV<->NVS bandwidth adds up to.
static int rcclTopoXmlFabricPeerCount(struct ncclXmlNode* xgmi) {
  struct ncclXmlNode* gpu = xgmi->parent;
  if (gpu == NULL || strcmp(gpu->name, "gpu") != 0) return 0;
  int total = 0;
  for (int s = 0; s < gpu->nSubs; s++) {
    struct ncclXmlNode* sub = gpu->subs[s];
    int fabricSupported = 0, count = 0;
    if (strcmp(sub->name, "xgmi") != 0) continue;
    if (xmlGetAttrIntDefault(sub, "fabric_supported", &fabricSupported, 0) != ncclSuccess || fabricSupported != 1) {
      continue;
    }
    if (xmlGetAttrIntDefault(sub, "count", &count, 0) == ncclSuccess) total += count;
  }
  return total;
}

// RCCL: DEV<->NVS link at fabric_bw_gbps once per DEV, else count * XGMI speed per fabric entry.
static ncclResult_t rcclTopoConnectFabricSwitch(struct ncclXmlNode* xgmi, struct ncclTopoNode* devNode,
                                                struct ncclTopoNode* nvs, int count) {
  float platformBw = 0;
  NCCLCHECK(xmlGetAttrFloatDefault(xgmi, "fabric_bw_gbps", &platformBw, 0.0f));
  bool linked = false;
  for (int l = 0; l < devNode->nlinks; l++) {
    if (devNode->links[l].remNode == nvs && devNode->links[l].type == LINK_NVL) linked = true;
  }
  const bool fromPlatform = std::isfinite(platformBw) && platformBw > 0;
  if (fromPlatform && linked) return ncclSuccess;

  float bw = platformBw;
  if (!fromPlatform) {
    const float nvlSpeed = ncclTopoXGMISpeed(devNode->dev.gcn);
    bw = count * nvlSpeed;
    if (!linked) {
      // The first entry also logs what the whole DEV will get.
      INFO(NCCL_GRAPH, "UALoE switch: DEV %lx-%lx <-> NVS %.1f GB/s per direction (default)",
           NCCL_TOPO_ID_SYSTEM_ID(devNode->id), NCCL_TOPO_ID_LOCAL_ID(devNode->id),
           std::max(rcclTopoXmlFabricPeerCount(xgmi), count) * nvlSpeed);
    }
  } else {
    INFO(NCCL_GRAPH, "UALoE switch: DEV %lx-%lx <-> NVS %.1f GB/s per direction (platform)",
         NCCL_TOPO_ID_SYSTEM_ID(devNode->id), NCCL_TOPO_ID_LOCAL_ID(devNode->id), bw);
  }
  NCCLCHECK(ncclTopoConnectNodes(devNode, nvs, LINK_NVL, bw));
  NCCLCHECK(ncclTopoConnectNodes(nvs, devNode, LINK_NVL, bw));
  return ncclSuccess;
}

// RCCL: XGMI is the AMD analogue of NVLink. Upstream v2.30 introduced the generic DEV
// (physical device) topology node, with GPU (rank) nodes hanging off their parent DEV node
// via LINK_LOC. ncclTopoComputePaths()/search now assemble inter-GPU links as DEV->DEV hops
// (see paths.cc: baseDevNode=gpu.parent, NVL links only between DEV nodes, and the
// GPU1-DEV1-DEV2-DEV3-GPU2 path shape). To stay consistent, XGMI links must be attached to
// the DEV nodes too (mirrors ncclTopoAddNvLinks below). XGMI link BW is keyed on the GCN
// arch string (not cudaCompCap), so we read it from devNode->dev.gcn, which ncclTopoAddPci
// populates from the device's first GPU rank.
ncclResult_t ncclTopoAddXGMI(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                             int systemId) {
  if (strcmp(node->name, "xgmi") == 0) {
    struct ncclTopoNode* devNode = NULL;
    if (node->parent && strcmp(node->parent->name, "gpu") == 0) {
      bool isPrimary = false;
      NCCLCHECK(ncclTopoXmlIsPrimaryGpuForDev(node->parent, &isPrimary));
      if (!isPrimary) return ncclSuccess;
      NCCLCHECK(ncclTopoGetGpuDevNode(node->parent, parentBusId, systemId, system, &devNode));
    } else {
      int64_t pBusId;
      NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
      NCCLCHECK(ncclTopoGetNode(system, &devNode, DEV, NCCL_TOPO_ID(systemId, pBusId)));
    }
    if (devNode == NULL) {
      WARN("Add XGMI error : could not find DEV for GPU %s\n", parentBusId);
      return ncclInternalError;
    }
    int count;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    const char* targetClass;
    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));
    int targetType;
    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));
    struct ncclTopoNode* remote = NULL;
    // RCCL: amd-smi lists only local fabric peers, so with several MNNVL hosts link every GPU's fabric
    // entries to one shared NVS; a single host keeps direct links.
    int fabricSupported = 0;
    NCCLCHECK(xmlGetAttrIntDefault(node, "fabric_supported", &fabricSupported, 0));
    bool viaFabricSwitch = false;
    if (targetType == GPU && fabricSupported == 1 && system->nHosts > 1) {
      NCCLCHECK(rcclTopoXmlIsCrossClique(node, &viaFabricSwitch));
    }
    if (targetType == GPU && !viaFabricSwitch) {
      // XGMI P2P connection to another GPU (between physical DEV nodes)
      const char* target;
      NCCLCHECK(xmlGetAttrStr(node, "target", &target));
      int64_t busId;
      NCCLCHECK(busIdToInt64(target, &busId));
      NCCLCHECK(ncclTopoGetDevForPciBusId(system, systemId, busId, &remote));
    } else if (targetType == CPU) {
      // XGMI connection to the local CPU
      NCCLCHECK(findLocalCpu(devNode, &remote, NULL));
    } else {
      if (system->nodes[NVS].count == 0) {
        NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));
      } else {
        remote = system->nodes[NVS].nodes;
      }
    }
    if (remote && viaFabricSwitch) {
      NCCLCHECK(rcclTopoConnectFabricSwitch(node, devNode, remote, count));
    } else if (remote) {
      float nvlSpeed = ncclTopoXGMISpeed(devNode->dev.gcn);
      NCCLCHECK(ncclTopoConnectNodes(devNode, remote, LINK_NVL, count * nvlSpeed));
      if (targetType != GPU) {
        NCCLCHECK(ncclTopoConnectNodes(remote, devNode, LINK_NVL, count * nvlSpeed));
      }
    }
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddXGMI(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}
#else
ncclResult_t ncclTopoAddNvLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                                int systemId) {
  if (strcmp(node->name, "nvlink") == 0) {
    bool isPrimary;
    NCCLCHECK(ncclTopoXmlIsPrimaryGpuForDev(node->parent, &isPrimary));
    if (!isPrimary) return ncclSuccess;
    struct ncclTopoNode* devNode = NULL;
    NCCLCHECK(ncclTopoGetGpuDevNode(node->parent, parentBusId, systemId, system, &devNode));
    if (devNode == NULL) {
      WARN("Add NVLink error : could not find DEV for GPU %s", parentBusId);
      return ncclInternalError;
    }
    int localDevsCount = 0;
    NCCLCHECK(ncclTopoGetDevNodes(system, devNode->id, NULL, &localDevsCount));
    int count, targetType;
    const char* targetClass;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));
    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));
    float nvlBw = ncclTopoNVLinkBw(devNode->dev.cudaCompCap);

    if (targetType == GPU) {
      const char* target;
      NCCLCHECK(xmlGetAttrStr(node, "target", &target));
      int64_t busId;
      NCCLCHECK(busIdToInt64(target, &busId));
      int remDevsCount = 0;
      struct ncclTopoNode* remDevs[NCCL_TOPO_MLOPART_DEV_MAX];
      NCCLCHECK(ncclTopoGetDevNodes(system, NCCL_TOPO_ID(systemId, busId), remDevs, &remDevsCount));
      // Bandwidth is split between the different devices both at source and destination.
      for (int j = 0; j < remDevsCount; j++) {
        NCCLCHECK(ncclTopoConnectNodes(devNode, remDevs[j], LINK_NVL, count * nvlBw / remDevsCount / localDevsCount));
      }
    } else {
      struct ncclTopoNode* remote = NULL;
      if (targetType == CPU) {
        NCCLCHECK(findLocalCpu(devNode, &remote, NULL));
      } else {
        if (system->nodes[NVS].count == 0) {
          NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));
        } else {
          remote = system->nodes[NVS].nodes;
        }
      }
      if (remote) {
        NCCLCHECK(ncclTopoConnectNodes(devNode, remote, LINK_NVL, count * nvlBw / localDevsCount));
        NCCLCHECK(ncclTopoConnectNodes(remote, devNode, LINK_NVL, count * nvlBw / localDevsCount));
      }
    }
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddNvLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}
#endif

ncclResult_t ncclTopoAddPciLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                                 int systemId) {
  if (strcmp(node->name, "pcilink") == 0) {
    struct ncclTopoNode* pci = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &pci, PCI, pBusId));
    if (pci == NULL) {
      WARN("Add PCI Link error : could not find PCI SW %lx", pBusId);
      return ncclInternalError;
    }
    struct ncclTopoNode* remote = NULL;
    const char* target;
    NCCLCHECK(xmlGetAttrStr(node, "target", &target));
    int64_t busId;
    NCCLCHECK(busIdToInt64(target, &busId));
    NCCLCHECK(ncclTopoGetNode(system, &remote, PCI, NCCL_TOPO_ID(systemId, busId)));
    if (remote) NCCLCHECK(ncclTopoConnectNodes(pci, remote, LINK_LOC, LOC_BW));
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddPciLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddC2c(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                            int systemId) {
  if (strcmp(node->name, "c2c") == 0) {
    bool isPrimary;
    NCCLCHECK(ncclTopoXmlIsPrimaryGpuForDev(node->parent, &isPrimary));
    if (!isPrimary) return ncclSuccess;
    struct ncclTopoNode* devNode = NULL;
    NCCLCHECK(ncclTopoGetGpuDevNode(node->parent, parentBusId, systemId, system, &devNode));
    if (devNode == NULL) {
      WARN("Add C2c error : could not find DEV for GPU %s", parentBusId);
      return ncclInternalError;
    }
    int nSibDevs = 0;
    NCCLCHECK(ncclTopoGetDevNodes(system, devNode->id, NULL, &nSibDevs));

    int linkCount = 0, bw = 0;
    NCCLCHECK(xmlGetAttrInt(node, "count", &linkCount));
    NCCLCHECK(xmlGetAttrInt(node, "bw", &bw));
    float c2cBw = (bw * linkCount) / 1000.0;
    struct ncclTopoNode* cpu = NULL;
    NCCLCHECK(findLocalCpu(devNode, &cpu, NULL));
    if (cpu == NULL) return ncclSuccess;

    if (nSibDevs > 1) {
      // Use a C2C bridge node to guarantee the total bw of the C2C link is shared between the devices.
      // Note: pBusId is the dev busId; we have checked that the mlopart bits are 0 in ncclTopoAddGpuSub
      int64_t xc2cId = devNode->id & ~(int64_t)NCCL_TOPO_MLOPART_MASK;
      struct ncclTopoNode* xc2cNode = NULL;
      NCCLCHECK(ncclTopoGetNode(system, &xc2cNode, CXB, xc2cId));
      if (xc2cNode == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &xc2cNode, CXB, xc2cId));
        NCCLCHECK(ncclTopoConnectNodes(xc2cNode, cpu, LINK_C2C, c2cBw));
        NCCLCHECK(ncclTopoConnectNodes(cpu, xc2cNode, LINK_C2C, c2cBw));
      }
      NCCLCHECK(ncclTopoConnectNodes(devNode, xc2cNode, LINK_C2C, c2cBw));
      NCCLCHECK(ncclTopoConnectNodes(xc2cNode, devNode, LINK_C2C, c2cBw));
    } else {
      NCCLCHECK(ncclTopoConnectNodes(devNode, cpu, LINK_C2C, c2cBw));
      NCCLCHECK(ncclTopoConnectNodes(cpu, devNode, LINK_C2C, c2cBw));
    }
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddC2c(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

// Rate one direction of a sibling hop at the on-package bw, whether or not the XGMI pass already
// built it. ncclTopoConnectNodes() accumulates onto an existing link of the same type, so the
// reported case has to be re-rated in place rather than topped up.
static ncclResult_t ncclTopoConnectMloPartSibling(struct ncclTopoNode* dev, struct ncclTopoNode* sibDev, float bw) {
  for (int l = 0; l < dev->nlinks; l++) {
    // No fixture can reach the LINK_NVL test: XGMI is the only link type any pass puts between two
    // DEV nodes, so the remNode test already excludes everything else a DEV carries (LINK_PCI to its
    // bridge, LINK_LOC to its GPU nodes, LINK_C2C to a CPU or C2C bridge). It stays because it picks
    // which link gets re-rated -- a later DEV-to-DEV link of another type would otherwise take the
    // on-package bw and leave the pair with no XGMI link, since the fallback below would not run.
    if (dev->links[l].remNode != sibDev || dev->links[l].type != LINK_NVL) continue;
    dev->links[l].bw = bw;
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "No XGMI link reported from MLOPart sibling %lx to %lx, assuming %.1f GB/s", dev->id, sibDev->id,
       bw);
  NCCLCHECK(ncclTopoConnectNodes(dev, sibDev, LINK_NVL, bw));
  return ncclSuccess;
}

// Compute partitions of one physical device (CPX/DPX, carried as MLOPart) reach each other over
// the on-package fabric, which sysfs reports as XGMI between the partitions' PCI functions, so
// ncclTopoAddXGMI() has normally already linked their DEV nodes. Cover the case where a platform
// does not report those links: without a link the partitions are unreachable. The hop is XGMI,
// not LINK_LOC -- PATH_LOC outranks PATH_NVL in the path search, so a LINK_LOC here would shadow
// the real XGMI links and hide the peer from everything keyed on PATH_NVL (P2P channel counts,
// XGMI topology classification), leaving it with the PCIe fallbacks.
//
// Rate every sibling hop at MLOPART_LOC_BW rather than at the width sysfs reports for the pair.
// The on-package fabric is far wider than the inter-device XGMI link whose width that report
// carries, and the search treats link bw as a budget: ncclTopoSearch's followPath() subtracts
// bwIntra from every link it lays a channel over. Rating a sibling hop at one XGMI width lets a
// partitioned node exhaust it, which walks the balanced-tree search down the whole speed array.
static ncclResult_t ncclTopoConnectMloPartSiblings(struct ncclTopoSystem* system) {
  for (int d = 0; d < system->nodes[DEV].count; d++) {
    struct ncclTopoNode* dev = system->nodes[DEV].nodes + d;
    struct ncclTopoNode* sibDevs[NCCL_TOPO_MLOPART_DEV_MAX];
    int nSibDevs = 0;
    NCCLCHECK(ncclTopoGetDevNodes(system, dev->id, sibDevs, &nSibDevs));
    for (int s = 0; s < nSibDevs; s++) {
      if (sibDevs[s] == dev) continue;
      NCCLCHECK(ncclTopoConnectMloPartSibling(dev, sibDevs[s], MLOPART_LOC_BW));
      NCCLCHECK(ncclTopoConnectMloPartSibling(sibDevs[s], dev, MLOPART_LOC_BW));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem,
                                      const uint64_t localHostHash) {
  NCCLCHECK(ncclCalloc(topoSystem, 1));
  struct ncclTopoSystem* system = *topoSystem;
  system->romeTopoModelIdx = RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE;
  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "system", &topNode));
  if (topNode == NULL) {
    WARN("ncclTopoGetSystemFromXml: system node not found in XML");
    return ncclInternalError;
  }
  for (int s = 0; s < topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "cpu") == 0) NCCLCHECK(ncclTopoAddCpu(node, *topoSystem));
  }

  int systemId = 0;
  while (systemId < system->nHosts && system->hostHashes[systemId] != localHostHash) systemId++;
  system->systemId = systemId;
  if (systemId == system->nHosts) {
    WARN("localHostHash = 0x%lx not found in the list of system hostHashes", localHostHash);
    return ncclInvalidArgument;
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  NCCLCHECK(ncclTopoAddXGMI(topNode, *topoSystem, NULL, 0));
#else
  NCCLCHECK(ncclTopoAddNvLinks(topNode, *topoSystem, NULL, 0));
#endif
  NCCLCHECK(ncclTopoAddC2c(topNode, *topoSystem, NULL, 0));
  NCCLCHECK(ncclTopoAddPciLinks(topNode, *topoSystem, NULL, 0));
  NCCLCHECK(ncclTopoConnectMloPartSiblings(*topoSystem));

  NCCLCHECK(ncclTopoFlattenBcmSwitches(*topoSystem));
  NCCLCHECK(ncclTopoConnectCpus(*topoSystem));
  NCCLCHECK(ncclTopoSortSystem(*topoSystem));

  return ncclSuccess;
}

NCCL_PARAM(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

static std::mutex netMutex;

ncclResult_t ncclTopoGetNetPropertiesSnapshot(struct ncclComm* comm, bool requireCommInitialized,
                                              struct ncclTopoNetPropertiesSnapshot** snapshots, int* nSnapshots) {
  ncclResult_t ret = ncclSuccess;
  int count = 0;

  if (comm == nullptr || snapshots == nullptr || nSnapshots == nullptr) {
    WARN("invalid arguments for network properties snapshot");
    return ncclInvalidArgument;
  }
  *snapshots = nullptr;
  *nSnapshots = 0;

  std::lock_guard<std::mutex> lock(netMutex);
  if (requireCommInitialized) {
    ncclResult_t initState = COMPILER_ATOMIC_LOAD(&comm->initState, std::memory_order_acquire);
    if (initState != ncclSuccess) return initState;
  }

  ncclNet_t* net = comm->ncclNet;
  if (net == nullptr) return ncclSuccess;
  if (net->devices == nullptr || net->getProperties == nullptr) {
    WARN("network plugin has invalid properties callbacks");
    return ncclInvalidArgument;
  }
  NCCLCHECKGOTO(net->devices(&count), ret, exit);
  if (count < 0) {
    WARN("network plugin returned invalid device count %d", count);
    ret = ncclInternalError;
    goto exit;
  }
  if (count == 0) goto exit;

  NCCLCHECKGOTO(ncclCalloc(snapshots, count), ret, exit);
  for (int dev = 0; dev < count; dev++) {
    struct ncclTopoNetPropertiesSnapshot* snapshot = *snapshots + dev;
    ncclNetProperties_t properties = {};

    if (net->getProperties(dev, &properties) != ncclSuccess) continue;
    snapshot->propertiesValid = true;
    snapshot->vProps = properties.vProps;
    snapshot->speed = properties.speed;
    snapshot->port = properties.port;
    snapshot->netDeviceType = properties.netDeviceType;
    if (properties.name != nullptr) {
      int n = snprintf(snapshot->name, sizeof(snapshot->name), "%s", properties.name);
      if (n < 0 || (size_t)n >= sizeof(snapshot->name)) snapshot->name[0] = '\0';
    }
    if (properties.pciPath != nullptr) {
      int n = snprintf(snapshot->pciPath, sizeof(snapshot->pciPath), "%s", properties.pciPath);
      if (n < 0 || (size_t)n >= sizeof(snapshot->pciPath)) snapshot->pciPath[0] = '\0';
    }
  }
  *nSnapshots = count;

exit:
  return ret;
}

struct ncclTopoXmlGatherGroup {
  struct ncclComm* comm;
  int* ranks;
  int rank;
  int nRanks;
};

static ncclResult_t ncclTopoBootstrapXmlAllGather(void* ctx, void* buf, size_t perRankSize) {
  struct ncclTopoXmlGatherGroup* group = (struct ncclTopoXmlGatherGroup*)ctx;
  return bootstrapIntraNodeAllGather(group->comm->bootstrap, group->ranks, group->rank, group->nRanks, buf,
                                     perRankSize);
}

ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) {
  ncclResult_t ret = ncclSuccess;
  struct ncclXml* xml = NULL;
  int* localRanks = NULL;
  int localRank = -1, nLocalRanks = 0;
  struct ncclTopoNetInfo nets[4] = {};
  int nNets = 0;
  struct ncclTopoXmlGatherGroup gatherGroup;
  struct ncclTopoDiscoveryConfig config = {};
  struct ncclGinState* ginState = &comm->ginState;
  ncclGin_t* gin = ginState->supported ? ginState->backends[0].ncclGin : NULL;
  ncclRma_t* rma = comm->rmaState.rmaProxyState.ncclRma;

  config.xmlTopoFile = ncclGetEnv("NCCL_TOPO_FILE");
  if (config.xmlTopoFile) INFO(NCCL_ENV, "NCCL_TOPO_FILE set by environment to %s", config.xmlTopoFile);
  config.dumpXmlFile = dumpXmlFile;
  config.busId = comm->peerInfo[comm->rank].busId;
  config.rank = comm->rank;
  config.gdrSupport = comm->peerInfo[comm->rank].gdrSupport;
  config.mloPart = comm->peerInfo[comm->rank].mloPart;
  config.minCompCap = comm->minCompCap;

  // net/gin/collnet share the same xml/graph nodes. Start with gin, then with collnet so that they take precedence.
  if (gin) {
    struct ncclTopoNetInfo* netInfo = nets + nNets++;
    netInfo->gin = 1;
    netInfo->netPluginIndex = ginState->backends[0].pluginIndex;
    netInfo->dmaBufSupport = comm->dmaBufSupport;
    netInfo->getDevCount = ncclGinGetDevCount;
    netInfo->name = gin->name;
    netInfo->getProperties = gin->getProperties;
    netInfo->devices = gin->devices;
  }
  if (rma) {
    struct ncclTopoNetInfo* netInfo = nets + nNets++;
    netInfo->rma = 1;
    netInfo->netPluginIndex = comm->rmaPluginIndex;
    netInfo->dmaBufSupport = comm->dmaBufSupport;
    netInfo->getDevCount = ncclRmaGetDevCount;
    netInfo->name = rma->name;
    netInfo->getProperties = rma->getProperties;
    netInfo->devices = rma->devices;
  }
  if (collNetSupport(comm)) {
    struct ncclTopoNetInfo* netInfo = nets + nNets++;
    netInfo->coll = 1;
    netInfo->netPluginIndex = comm->netPluginIndex;
    netInfo->maxDevsPerNic = (comm->ncclNetVer >= 12) ? NCCL_NET_MAX_DEVS_PER_NIC : NCCL_NET_MAX_DEVS_PER_NIC_V11;
    netInfo->dmaBufSupport = comm->dmaBufSupport;
    netInfo->getDevCount = ncclCollNetGetDevCount;
    netInfo->setVirtDevCount = ncclCollNetSetVirtDevCount;
    netInfo->name = comm->ncclCollNet->name;
    netInfo->getProperties = comm->ncclCollNet->getProperties;
    netInfo->makeVDevice = comm->ncclCollNet->makeVDevice;
    netInfo->devices = comm->ncclCollNet->devices;
  }
  {
    struct ncclTopoNetInfo* netInfo = nets + nNets++;
    netInfo->net = 1;
    netInfo->netPluginIndex = comm->netPluginIndex;
    netInfo->maxDevsPerNic = (comm->ncclNetVer >= 12) ? NCCL_NET_MAX_DEVS_PER_NIC : NCCL_NET_MAX_DEVS_PER_NIC_V11;
    netInfo->dmaBufSupport = comm->dmaBufSupport;
    netInfo->getDevCount = ncclNetGetDevCount;
    netInfo->setVirtDevCount = ncclNetSetVirtDevCount;
    netInfo->name = comm->ncclNet->name;
    netInfo->getProperties = comm->ncclNet->getProperties;
    netInfo->makeVDevice = comm->ncclNet->makeVDevice;
    netInfo->devices = comm->ncclNet->devices;
  }
  config.nets = nets;
  config.nNets = nNets;
  config.netLock = &netMutex;

  // XML topo fusion.
  if (comm->MNNVL) {
    // MNNVL clique support
    nLocalRanks = comm->clique.size;
    localRank = comm->cliqueRank;
    localRanks = comm->clique.ranks;
  } else {
    // Intra-node fusion.  Much of the comm is not initialized yet at this point so we need to do our own calculations.
    NCCLCHECKGOTO(ncclCalloc(&localRanks, comm->nRanks), ret, fail);
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank) localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
  }
  gatherGroup.comm = comm;
  gatherGroup.ranks = localRanks;
  gatherGroup.rank = localRank;
  gatherGroup.nRanks = nLocalRanks;
  config.nLocalRanks = nLocalRanks;
  config.localRank = localRank;
  config.allGather = ncclTopoBootstrapXmlAllGather;
  config.allGatherCtx = &gatherGroup;
  config.crossClique = comm->MNNVL && comm->clique.size < comm->nRanks;

  NCCLCHECKGOTO(ncclTopoDiscoverXml(&config, &xml), ret, fail);

  if (dumpXmlFile && comm->rank == ncclParamTopoDumpFileRank()) {
    INFO(NCCL_ENV, "NCCL_TOPO_DUMP_FILE set by environment to %s", dumpXmlFile);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(dumpXmlFile, xml), ret, fail);
  }

  // Only update our topo tracking structure if we aren't dumping (separate steps)
  if (dumpXmlFile == NULL) {
    NCCLCHECKGOTO(ncclTopoGetSystemFromXml(xml, system, getHostHash()), ret, fail);
    // store comm related information
    (*system)->minDriverVersion = comm->minDriverVersion;
    (*system)->cuMemGdrSupport = comm->cuMemGdrSupport;
  }

exit:
  if (!comm->MNNVL) free(localRanks);
  free(xml);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                              int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType) {
  int minType = PATH_DIS;
  float maxBw = 0;
  int count = 0;
  struct ncclTopoLinkList* paths = system->nodes[type].nodes[index].paths[resultType];
  if (paths == NULL) {
    *localCount = 0;
    return ncclSuccess;
  }
  for (int i = 0; i < system->nodes[resultType].count; i++) {
    if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
      maxBw = paths[i].bw;
      minType = paths[i].type;
      if (pathType) *pathType = minType;
      count = 0;
    }
    if (paths[i].bw == maxBw && paths[i].type == minType) {
      if (count == NCCL_TOPO_MAX_NODES) {
        WARN("Error : ran out of room to store found nodes in ncclTopoGetLocal."
             " Filled %d of type %d, starting from index %d of type %d.",
             NCCL_TOPO_MAX_NODES, resultType, index, type);
        return ncclInternalError;
      }
      locals[count++] = i;
    }
  }
  *localCount = count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw) {
  // Assuming BW to CPU reflects the GPU bandwidth via P2P or C2C.
  // Caveat, this could be wrong if there is a PCIe switch, and a narrower link to the CPU.
  int c;
  NCCLCHECK(ncclGetLocalCpu(system, gpu, &c));
  float gpuBw = system->nodes[GPU].nodes[gpu].paths[CPU][c].bw;
  int rank = system->nodes[GPU].nodes[gpu].gpu.rank;

  // RCCL: on configurations with no local network device (e.g. a single-node
  // run using the socket transport with no IB devices), the topology has no
  // GPU->NET path. Report zero net devices instead of failing, matching how
  // callers (paths.cc) already handle a zero count.
  {
    int localNets[NCCL_TOPO_MAX_NODES];
    int localNetCount = 0;
    NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, NET, localNets, &localNetCount, NULL));
    if (localNetCount == 0) {
      *count = 0;
      *bw = 0;
      return ncclSuccess;
    }
  }

  int netCountByBw = 0;
  float totalNetBw = 0;
  int64_t firstNetId = 0;
  for (int c = 0; c < MAXCHANNELS; c++) {
    int net;
    int64_t netId;
    NCCLCHECK(ncclTopoGetLocalNet(system, rank, c, &netId, NULL));
    NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &net));
    if (c == 0) firstNetId = netId;
    else if (firstNetId == netId) break;

    totalNetBw += system->nodes[GPU].nodes[gpu].paths[NET][net].bw;
    netCountByBw++;
    if (totalNetBw >= gpuBw) break;
  }
  *count = netCountByBw;
  *bw = totalNetBw;
  return ncclSuccess;
}

static int netDevsPolicyNum = -1;
static enum netDevsPolicy netDevsPolicy = NETDEVS_POLICY_UNDEF;
static void getNetDevsPolicyOnce() {
  const char* envStr = ncclGetEnv("NCCL_NETDEVS_POLICY");
  if (envStr) {
    if (strcasecmp(envStr, "AUTO") == 0) {
      netDevsPolicy = NETDEVS_POLICY_AUTO;
    } else if (strcasecmp(envStr, "ALL") == 0) {
      netDevsPolicy = NETDEVS_POLICY_ALL;
    } else if (strncasecmp(envStr, "MAX:", strlen("MAX:")) == 0) {
      int envNum = atoi(envStr + strlen("MAX:"));
      if (envNum > 0) {
        netDevsPolicy = NETDEVS_POLICY_MAX;
        netDevsPolicyNum = envNum;
      }
    }
    if (netDevsPolicy == NETDEVS_POLICY_UNDEF) {
      INFO(NCCL_ENV, "Unable to recognize NCCL_NETDEVS_POLICY=%s, using NCCL_NETDEVS_POLICY_AUTO instead.", envStr);
    } else {
      INFO(NCCL_ENV, "NCCL_NETDEVS_POLICY set by environment to %s", envStr);
    }
  }
  if (netDevsPolicy == NETDEVS_POLICY_UNDEF) netDevsPolicy = NETDEVS_POLICY_AUTO;
}

ncclResult_t ncclTopoGetNetDevsPolicy(enum netDevsPolicy* policy, int* policyNum) {
  static std::once_flag onceFlag;
  std::call_once(onceFlag, getNetDevsPolicyOnce);
  if (netDevsPolicy == NETDEVS_POLICY_MAX && netDevsPolicyNum <= 0) {
    WARN("Invalid number of network devices = %d for policy MAX", netDevsPolicyNum);
    return ncclInternalError;
  }
  if (policy) *policy = netDevsPolicy;
  if (policyNum && netDevsPolicyNum >= 0) *policyNum = netDevsPolicyNum;
  return ncclSuccess;
}

// Sort localNets by (rail, plane); rail r spans [railOffset[r], railOffset[r+1]).
struct netRailSort {
  int netIndex;
  int railIdx;
  int planeId;
  int idx;
};

static int compareRailPlane(const void* a, const void* b) {
  const struct netRailSort* x = (const struct netRailSort*)a;
  const struct netRailSort* y = (const struct netRailSort*)b;
  if (x->railIdx != y->railIdx) return (x->railIdx < y->railIdx) ? -1 : 1;
  if (x->planeId != y->planeId) return (x->planeId < y->planeId) ? -1 : 1;
  if (x->idx != y->idx) return (x->idx < y->idx) ? -1 : 1;
  return 0;
}

static ncclResult_t ncclTopoSortDevsByRailPlane(struct ncclTopoSystem* system, int type, int* localNets,
                                                int localNetCount, int* railOffset, int* localRailCount) {
  *localRailCount = 0;
  if (localNetCount == 0) return ncclSuccess;

  struct netRailSort sorted[NCCL_TOPO_MAX_NODES];
  for (int d = 0; d < localNetCount; d++) {
    struct ncclTopoNode* netNode = system->nodes[type].nodes + localNets[d];
    int idx = 0;
    while (idx < d) {
      struct ncclTopoNode* tmpNode = system->nodes[type].nodes + localNets[idx];
      if (tmpNode->net.railId == netNode->net.railId) break;
      idx++;
    }
    sorted[d].railIdx = idx;
    sorted[d].netIndex = localNets[d];
    sorted[d].planeId = netNode->net.planeId;
    sorted[d].idx = d;
  }
  qsort(sorted, localNetCount, sizeof(struct netRailSort), compareRailPlane);

  railOffset[0] = 0;
  *localRailCount = 1;
  for (int d = 0; d < localNetCount; d++) {
    localNets[d] = sorted[d].netIndex;
    if (d > 0 && sorted[d].railIdx != sorted[d - 1].railIdx) (*localRailCount)++;
    railOffset[*localRailCount] = d + 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalNetType(struct ncclTopoSystem* system, int type, int rank, int channelId, int64_t* id,
                                     int* dev) {
  int gpu;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpu, /*showWarn=*/true));

  int localNets[NCCL_TOPO_MAX_NODES];
  int localNetCount;
  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, type, localNets, &localNetCount, NULL));
  if (localNetCount == 0) {
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
    WARN("Could not find any local path from gpu %d to net.", gpu);
#endif
    return ncclInternalError;
  }

  int localGpuCount = 0, netsPerGpu = 0, policyCount = 0, localRailCount = 0;
  int localGpus[NCCL_TOPO_MAX_NODES];
  int railOffset[NCCL_TOPO_MAX_NODES + 1];
  enum netDevsPolicy policy;
  NCCLCHECK(ncclTopoGetNetDevsPolicy(&policy, &policyCount));
  NCCLCHECK(ncclTopoGetLocal(system, type, localNets[0], GPU, localGpus, &localGpuCount, NULL));

  struct ncclTopoNode* gpuNode = system->nodes[GPU].nodes + gpu;
  int net = 0;
  NCCLCHECK(ncclTopoSortDevsByRailPlane(system, type, localNets, localNetCount, railOffset, &localRailCount));
  if (localRailCount == 0) {
    WARN("No rails found for GPU/%d.", gpuNode->gpu.dev);
    return ncclInternalError;
  }
  int rail = 0;
  int devIdx = gpuNode->gpu.dev;
  if (isPow2(localRailCount)) {
    rail = mirrorBits(devIdx, localRailCount);
  } else {
    rail = (gpuNode->gpu.dev * DIVUP(localRailCount, localGpuCount)) % localRailCount;
  }
  int railEnd = rail + DIVUP(localRailCount, localGpuCount);
  netsPerGpu = railOffset[railEnd % localRailCount] + ((railEnd < localRailCount) ? 0 : railOffset[localRailCount]) -
               railOffset[rail];
  net = railOffset[rail];
  // overwrite the netsPerGpu if needed
  if (policy == NETDEVS_POLICY_ALL) {
    netsPerGpu = localNetCount;
  } else if (policy == NETDEVS_POLICY_MAX) {
    netsPerGpu = std::min(policyCount, localNetCount);
  }

  net += channelId % netsPerGpu;
  if (id) *id = system->nodes[type].nodes[localNets[net % localNetCount]].id;
  if (dev) *dev = system->nodes[type].nodes[localNets[net % localNetCount]].net.dev;
  return ncclSuccess;
}
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  return ncclTopoGetLocalNetType(system, NET, rank, channelId, id, dev);
}
ncclResult_t ncclTopoGetLocalGinDev(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  return ncclTopoGetLocalNetType(system, GIN, rank, channelId, id, dev);
}

ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm* comm, int* localGinDevs, int* localGinCount) {
  for (int c = 0; c < NCCL_TOPO_MAX_NODES; c++) {
    NCCLCHECK(ncclTopoGetLocalGinDev(comm->topo, comm->rank, c, NULL, localGinDevs + c));
    if (c > 0 && localGinDevs[c] == localGinDevs[0]) {
      *localGinCount = c;
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalRmaDev(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  return ncclTopoGetLocalNetType(system, RMA, rank, channelId, id, dev);
}

ncclResult_t ncclTopoGetLocalRmaDevs(struct ncclComm* comm, int* localRmaDevs, int* localRmaCount) {
  for (int c = 0; c < NCCL_TOPO_MAX_NODES; c++) {
    NCCLCHECK(ncclTopoGetLocalRmaDev(comm->topo, comm->rank, c, NULL, localRmaDevs + c));
    if (c > 0 && localRmaDevs[c] == localRmaDevs[0]) {
      *localRmaCount = c;
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex) {
  ncclResult_t ret = ncclSuccess;
  int netIndex;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &netIndex));

  int localGpus[NCCL_TOPO_MAX_NODES];
  int localGpuCount;
  NCCLCHECK(ncclTopoGetLocal(system, NET, netIndex, GPU, localGpus, &localGpuCount, NULL));

  int foundGpu = -1;
  for (int c = 0; c < MAXCHANNELS; c++) {
    for (int lg = 0; lg < localGpuCount; lg++) {
      int g = localGpus[lg];
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
      int64_t id;
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &id, NULL));
      if (netId == id) {
        foundGpu = g;
        goto exit;
      }
    }
  }
exit:
  *gpuIndex = foundGpu;
  return ret;
}

/****************************/
/* External query functions */
/****************************/

ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model) {
  *arch = system->nodes[CPU].nodes[0].cpu.arch;
  *vendor = system->nodes[CPU].nodes[0].cpu.vendor;
  *model = system->nodes[CPU].nodes[0].cpu.model;
  return ncclSuccess;
}

bool rcclTopoCpuIsRome(struct ncclTopoSystem* system) {
  const struct ncclTopoNode* cpu = system->nodes[CPU].nodes;
  return cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD && cpu->cpu.rome;
}

NCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity) {
  struct ncclTopoNode *cpu = NULL, *gpu = NULL;
  int gpuIndex, cpuIndex;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpuIndex, /*showWarn=*/true));
  NCCLCHECK(ncclGetLocalCpu(system, gpuIndex, &cpuIndex));
  gpu = system->nodes[GPU].nodes + gpuIndex;
  cpu = system->nodes[CPU].nodes + cpuIndex;

  // Query the CPU affinity set we were provided
  ncclAffinity mask;
  NCCLCHECK(ncclOsGetAffinity(&mask));

  // Get the affinity of the CPU close to our GPU.
  ncclAffinity cpuMask = cpu->cpu.affinity;

  // Get the final affinity
  ncclAffinity finalMask;
  if (ncclParamIgnoreCpuAffinity()) {
    // Ignore the CPU affinity set and use the GPU one instead
    finalMask = cpuMask;
  } else {
    // Use a subset of the GPU affinity set
    finalMask = ncclOsCpuAnd(mask, cpuMask);
  }

  memcpy(affinity, &finalMask, sizeof(ncclAffinity));

  // display the final affinity
  char msg[1024] = "";
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "Affinity for GPU %d is ", gpu->gpu.dev);
  if (ncclOsCpuCount(finalMask)) {
    (void)ncclCpusetToRangeStr(&finalMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  } else {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "empty, ignoring");
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ". (GPU affinity = ");
  (void)ncclCpusetToRangeStr(&cpuMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  if (!ncclParamIgnoreCpuAffinity()) {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " ; CPU affinity = ");
    (void)ncclCpusetToRangeStr(&mask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ").");
  INFO(NCCL_INIT, "%s: %s", __func__, msg);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[GPU].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NET].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NVS].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax) {
  if (system->nodes[DEV].count == 0) return ncclInternalError;
  int min, max;
  min = max = system->nodes[DEV].nodes[0].dev.cudaCompCap;
  for (int d = 1; d < system->nodes[DEV].count; d++) {
    min = std::min(min, system->nodes[DEV].nodes[d].dev.cudaCompCap);
    max = std::max(max, system->nodes[DEV].nodes[d].dev.cudaCompCap);
  }
  if (ccMin) *ccMin = min;
  if (ccMax) *ccMax = max;
  return ncclSuccess;
}

int64_t ncclParamMultiRankGpuEnable();

ncclResult_t ncclCheckMultiRank(struct ncclComm* comm) {
  // Check if any ranks on this node share the same GPU
  // It would be nice to know the local ranks, but that information
  // may not yet be available and we want to disable nvls before it is
  // initialized.
  int rank = comm->rank;
  const int MAX_RANKS_ON_HOST = 2048;
  int64_t gpuFoundBusIds[MAX_RANKS_ON_HOST];
  int gpuFoundRanks[MAX_RANKS_ON_HOST];
  int numRanksFound = 0;
  int firstRankOnHost = -1;
  comm->isMultiRankGpu = false;

  for (int i = 0; i < comm->nRanks; i++) {
    if (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) {
      if (firstRankOnHost == -1) firstRankOnHost = i;
      int64_t bus = comm->peerInfo[i].busId;
      for (int j = 0; j < numRanksFound; ++j) {
        if (bus == gpuFoundBusIds[j]) {
          comm->isMultiRankGpu = true;
          if (rank == firstRankOnHost) {
            INFO(NCCL_INIT, "Detected ranks %d and %d sharing GPU with bus %lX on this node", gpuFoundRanks[j], i, bus);
          }
        }
      }
      if (numRanksFound == MAX_RANKS_ON_HOST) {
        if (firstRankOnHost == rank) {
          WARN("ncclCheckMultiRank exceeded number of local ranks %d", MAX_RANKS_ON_HOST);
        }
        return ncclInternalError;
      }
      gpuFoundBusIds[numRanksFound] = bus;
      gpuFoundRanks[numRanksFound++] = i;
    }
  }

  if ((firstRankOnHost == -1 || firstRankOnHost == rank) && comm->isMultiRankGpu &&
      ncclParamMultiRankGpuEnable() == 0) {
    WARN("Multiple ranks detected using the same GPU on this node."
         " Set NCCL_MULTI_RANK_GPU_ENABLE=1 to enable this"
         " configuration.");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}
