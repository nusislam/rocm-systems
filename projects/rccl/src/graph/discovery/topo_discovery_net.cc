/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Network device import: lists the devices of each network provider in the topology XML, and fuses physical NICs
// into virtual ones.

#include "core.h"
#include "topo_discovery.h"
#include "../topo.h"
#include <algorithm>
#include <cstring>
#include <mutex>

static ncclResult_t ncclTopoGetNetDevKey(struct ncclXmlNode* xmlNet, uint64_t* key) {
  // Create the unique key per device to be used for rail logic.
  // Note: virtual device might have the same key as their first physical device.
  uint64_t hacc[2] = {1, 1}, guid;
  NCCLCHECK(xmlGetAttrUint64Default(xmlNet, "guid", &guid, 0));
  eatHash(hacc, &guid);
  *key = digestHash(hacc);
  return ncclSuccess;
}

// return the index of the rail key in the unique rail key table
static ncclResult_t ncclTopoGetNetRailKeyIndex(struct ncclTopoNetRailKeyList* list, uint64_t key, int* railId) {
  auto it = list->keys.find(key);
  if (it != list->keys.end()) {
    *railId = it->second;
    return ncclSuccess;
  }
  if (list->keys.size() == NCCL_TOPO_MAX_NODES) {
    WARN("NCCL cannot use more than %d rails.", NCCL_TOPO_MAX_NODES);
    return ncclInternalError;
  }
  *railId = list->keys.size();
  list->keys[key] = *railId;
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetRailPlane(struct ncclTopoNetRailKeyList* list, struct ncclXmlNode* xmlNet, int* rail,
                                         int* plane) {
  if (*rail == NCCL_TOPO_UNDEF) {
    uint64_t railKey;
    int keyIndex;
    NCCLCHECK(ncclTopoGetNetDevKey(xmlNet, &railKey));
    NCCLCHECK(ncclTopoGetNetRailKeyIndex(list, railKey, &keyIndex));
    *rail = NCCL_TOPO_UNDEF_BIT | keyIndex;
  }
  if (*plane == NCCL_TOPO_UNDEF) {
    // Planes are obtained as the index of this device within its PCI device.
    // We discard devices with planes provided by the plugin and devices of different type.
    int count = 0;
    struct ncclXmlNode* nic = xmlNet->parent;
    for (int s = 0; nic && s < nic->nSubs; s++) {
      if (xmlNet == nic->subs[s]) break;
      int subPlane = NCCL_TOPO_UNDEF;
      NCCLCHECK(xmlGetAttrIntDefault(nic->subs[s], "plane", &subPlane, NCCL_TOPO_UNDEF));
      if (strcmp(xmlNet->name, nic->subs[s]->name) == 0 &&
          (subPlane == NCCL_TOPO_UNDEF || (subPlane & NCCL_TOPO_UNDEF_BIT)))
        count++;
    }
    *plane = NCCL_TOPO_UNDEF_BIT | count;
  }
  return ncclSuccess;
}

// This is just checking for direct descendence
int ncclTopoCheckPix(ncclXmlNode* common, ncclXmlNode** nodes, int nNodes) {
  const char* tempBusId;
  // If the common parent isn't a pci switch, then this isn't PIX
  NCCLCHECK(xmlGetAttrStr(common, "busid", &tempBusId));
  if (tempBusId == NULL) return 0;
  TRACE(NCCL_GRAPH, "Checking pix for busid=%s", tempBusId);

  // All the nodes must have a "nic" which is a parent, and then a pci node (busid) which must be a child of the
  // "common"
  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* node = nodes[i];
    if (strcmp(node->name, "net") == 0) {
      node = node->parent;
      if (node == NULL) return 0;
      if (strcmp(node->name, "nic") == 0) {
        node = node->parent;
        if (node == NULL) return 0;
        // All nodes must descend from the same first level pci switch
        if (strcmp(node->name, "pci") == 0) {
          TRACE(NCCL_GRAPH, "Comparing parent of node=%p to common=%p", node->parent, common);
          if (node->parent != common) return 0;
        }
      }
    }
  }

  return 1;
}

#define NCCL_TOPO_XML_DEPTH_MAX 256
typedef struct xmlNodeStack {
  ncclXmlNode* elems[NCCL_TOPO_XML_DEPTH_MAX];
  int tail;

  ncclXmlNode* top() {
    if (!empty()) {
      return elems[tail - 1];
    } else {
      return NULL;
    }
  }

  ncclXmlNode* pop() {
    ncclXmlNode* node = top();
    if (node) {
      tail--;
    }
    return node;
  }

  void push(ncclXmlNode* node) {
    if (tail < NCCL_TOPO_XML_DEPTH_MAX) {
      elems[tail++] = node;
    }
  }

  bool empty() {
    return tail == 0;
  }

} xmlNodeStack;

ncclResult_t ncclFindFirstPciParent(ncclXmlNode** parent) {
  ncclXmlNode* newParent = *parent;
  while (strcmp(newParent->name, "pci") != 0) {
    newParent = newParent->parent;
    if (newParent == nullptr) return ncclSuccess;
    if (strcmp(newParent->name, "system") == 0) return ncclSuccess;
  }
  *parent = newParent;
  return ncclSuccess;
}

// 1. Find the common parent xmlNode between the given set of nodes
ncclResult_t ncclTopoGetPath(ncclXmlNode** nodes, int nNodes, int* path, ncclXmlNode** parent) {
  // Track a stack of parents per-net node being merged
  xmlNodeStack* parents;
  NCCLCHECK(ncclCalloc(&parents, nNodes));
  // Find the common parent
  ncclXmlNode* common = NULL;

  if (nNodes == 1) {
    common = nodes[0];
    *path = PATH_LOC;
    goto out;
  }

  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* temp;
    temp = nodes[i];
    while (temp) {
      parents[i].push(temp);
      temp = strcmp(temp->name, "system") == 0 ? NULL : temp->parent;
    }
  }

  common = NULL;
  int c;
  c = 1;
  while (c && !parents[0].empty()) {
    ncclXmlNode* temp = parents[0].top();
    for (int i = 1; i < nNodes; i++) {
      if (!parents[i].empty()) {
        c &= (temp == parents[i].top());
      } else {
        c = 0;
        break;
      }
    }

    if (c) {
      common = temp;
      if (common == NULL) {
        TRACE(NCCL_GRAPH, "COMMON IS NULL");
      }
      for (int i = 0; i < nNodes; i++) {
        parents[i].pop();
      }
      // Check multi-port while we still have the mismatched parents
      // For multi-port to be true, all parents (peers) must have the busId attribute with all but the last character matching
    } else {
      int multiPort = 1;
      const char* tempBusId;

      NCCLCHECK(xmlGetAttr(temp, "busid", &tempBusId));
      if (tempBusId) {
        for (int i = 1; i < nNodes; i++) {
          if (!parents[i].empty()) {
            const char* busId;
            NCCLCHECK(xmlGetAttr(parents[i].top(), "busid", &busId));
            if (busId) {
              if (strlen(busId) != strlen(tempBusId)) {
                multiPort = 0;
                break;
              }
              if (strncmp(busId, tempBusId, strlen(busId) - 1) != 0) {
                multiPort = 0;
                break;
              }
            } else {
              multiPort = 0;
              break;
            }
          }
        }
      } else {
        multiPort = 0;
      }

      if (multiPort) {
        *path = PATH_PORT;
        goto out;
      }
    }
  }

  if (common == NULL) {
    *path = PATH_DIS;
  } else if (strcmp(common->name, "system") == 0) {
    *path = PATH_SYS;
  } else if (strcmp(common->name, "cpu") == 0) {
    *path = PATH_PHB;
  } else if (strcmp(common->name, "nic") == 0) {
    *path = PATH_PORT;
  } else if (strcmp(common->name, "net") == 0) {
    *path = PATH_PORT;
  } else if (ncclTopoCheckPix(common, nodes, nNodes)) {
    *path = PATH_PIX;
  } else {
    *path = PATH_PXB;
  }

out:
  ncclFindFirstPciParent(&common);
  *parent = common;
  free(parents);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeUniqueBusId(struct ncclXml* xml, char* busId, struct ncclXmlNode** pciNode,
                                     struct ncclXmlNode* parent) {
  int i = 0;
  int64_t rBusId;
  NCCLCHECK(busIdToInt64(busId, &rBusId));
  // Try to find an unused busid - NCCL expects leaf busid to be unique
  while (i < 100) {
    rBusId++;
    TRACE(NCCL_GRAPH, "Trying to make new busId %lx", rBusId);
    int64ToBusId(rBusId, busId);
    struct ncclXmlNode* temp = NULL;
    NCCLCHECK(xmlFindTagKv(xml, "pci", &temp, "busid", busId));
    if (temp == NULL) {
      NCCLCHECK(xmlAddNode(xml, parent, "pci", pciNode));
      NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
      TRACE(NCCL_GRAPH, "Made new busId %lx", rBusId);
      return ncclSuccess;
    }
    TRACE(NCCL_GRAPH, "Conflicting busId %lx", rBusId);
    i++;
  }

  WARN("TOPO/NET : Couldn't generate unique busId after %d tries", i);
  return ncclInternalError;
}

// Add a new PCI node with a unique busId under (*parent) and overwrite the value of (*parent) to point to the new node
ncclResult_t ncclTopoMakePciParent(struct ncclXml* xml, struct ncclXmlNode** parent, struct ncclXmlNode* physNetNode) {
  struct ncclXmlNode* newBusId = NULL;
  struct ncclXmlNode* pci = physNetNode->parent;
  if (pci) {
    pci = pci->parent;
    if (pci) {
      if (strcmp(pci->name, "pci") == 0) {
        char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        memset(busId, 0, sizeof(busId));
        const char* originalBusId;
        // Seed busId with the current NIC 0's busId to make discovering a unique hash quicker
        NCCLCHECK(xmlGetAttrStr(pci, "busid", &originalBusId));
        snprintf(busId, sizeof(busId), "%s", originalBusId);
        NCCLCHECK(ncclTopoMakeUniqueBusId(xml, busId, &newBusId, *parent));
        for (int i = 0; i < pci->nAttrs; i++) {
          NCCLCHECK(xmlSetAttr(newBusId, pci->attrs[i].key, pci->attrs[i].value));
        }
        NCCLCHECK(xmlSetAttr(newBusId, "busid", busId));
        *parent = newBusId;
      }
    }
  }

  if (newBusId == NULL) {
    const char* name;
    NCCLCHECK(xmlGetAttr(physNetNode, "name", &name));
    WARN("TOPO/NET : Can't find busId of child 0 %s", name);
    return ncclInternalError;
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoMakeVnic(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, ncclNetVDeviceProps_t* vProps,
                              struct ncclXmlNode** physNetNodes) {
  if (vProps->ndevs > netInfo->maxDevsPerNic) {
    WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps->ndevs, netInfo->maxDevsPerNic);
    return ncclInternalError;
  }

  // Don't make vNics of size 1
  if (vProps->ndevs == 1) {
    TRACE(NCCL_GRAPH, "TOPO/NET : Skipping vNic of size 1");
    return ncclSuccess;
  }

  // Trigger the merge, then get the new device's properties
  int vDevIndex = 0;
  ncclResult_t ret;
  NOWARN(ret = netInfo->makeVDevice(&vDevIndex, vProps), NCCL_GRAPH | NCCL_INIT | NCCL_NET);
  if (ret != ncclSuccess) {
    INFO(NCCL_GRAPH | NCCL_INIT | NCCL_NET,
         "TOPO/NET : Tried merging multiple devices together and failed. vProps={ndevs=%d, devs=[%d %d %d %d]}. Set "
         "NCCL_NET_MERGE_LEVEL=LOC to disable NIC fusion.",
         vProps->ndevs, vProps->devs[0], vProps->devs[1], vProps->devs[2], vProps->devs[3]);
    return ret;
  }

  INFO(NCCL_GRAPH, "TOPO/NET : Made vNic %d", vDevIndex);
  return ncclSuccess;
}

ncclResult_t ncclTopoForceMerge(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int* placedDevs,
                                ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs) {
  ncclResult_t ret = ncclSuccess;
  const char* str = netInfo->forceMerge;
  INFO(NCCL_ENV | NCCL_NET, "TOPO/NET : Force-fusing NICs using NCCL_NET_FORCE_MERGE=%s", str);
  char* ncStr;
  NCCLCHECK(ncclCalloc(&ncStr, strlen(str) + 1));
  strcpy(ncStr, str);
  char* semi_token;
  char* semi = strtok_r(ncStr, ";", &semi_token);
  while (semi) {
    TRACE(NCCL_NET, "Fusing %s", semi);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // RCCL: one entry past the limit. parseStringList() truncates silently, so without it an
    // over-limit group is indistinguishable from a legal full-size one.
    const int maxUserIfs = NCCL_NET_MAX_DEVS_PER_NIC + 1;
#else
    const int maxUserIfs = NCCL_NET_MAX_DEVS_PER_NIC;
#endif
    struct netIf userIfs[maxUserIfs];
    int nUserIfs = parseStringList(semi, userIfs, maxUserIfs);
    if (nUserIfs == 0) {
      INFO(NCCL_NET,
           "NET/IB : Invalid NCCL_NET_FORCE_MERGE specified %s. Couldn't parse substring %s. Please provide a "
           "semicolon-delimited list of comma-delimited NIC groups.",
           ncStr, semi);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
      // RCCL: the loop advances semi at the bottom, so continuing without a step of its own
      // re-parses the same unparsable substring forever. ":" and "," both parse to zero NICs.
      semi = strtok_r(NULL, ";", &semi_token);
#endif
      continue;
    }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // RCCL: reject a group that requests more NICs than one vNIC can hold, see maxUserIfs above.
    // The count is a lower bound, since parsing stopped one entry past the limit.
    if (nUserIfs > NCCL_NET_MAX_DEVS_PER_NIC) {
      WARN("TOPO/NET : Specified fused NIC %s which requests at least %d devices. Max %d", semi, nUserIfs,
           NCCL_NET_MAX_DEVS_PER_NIC);
      ret = ncclInvalidUsage;
      goto fail;
    }
#endif

    ncclNetVDeviceProps_t vProps = {0};
    for (int d = 0; d < nPhysDevs; d++) {
      if (matchIfList(propsList[d].name, propsList[d].port, userIfs, nUserIfs, 1)) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        // RCCL: a pattern given without ":port" matches every port of a multi-port NIC, so the
        // number of matches is not bounded by nUserIfs and can overrun devs[].
        if (vProps.ndevs == NCCL_NET_MAX_DEVS_PER_NIC) {
          WARN("TOPO/NET : Specified fused NIC %s which matches too many devices. Max %d", semi,
               NCCL_NET_MAX_DEVS_PER_NIC);
          ret = ncclInvalidUsage;
          goto fail;
        }
#endif
        vProps.devs[vProps.ndevs++] = d;
      }
    }

    if (vProps.ndevs != nUserIfs) {
      WARN("TOPO/NET : Only matched %d devices, %d requested from %s", vProps.ndevs, nUserIfs, semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    if (vProps.ndevs > netInfo->maxDevsPerNic) {
      WARN("Specified fused NIC %s which has too many devices (%d). Max %d", semi, vProps.ndevs,
           netInfo->maxDevsPerNic);
      ret = ncclInvalidUsage;
      goto fail;
    }

    ret = ncclTopoMakeVnic(xml, netInfo, &vProps, physNetNodes);
    if (ret == ncclSuccess) {
      // Only set that a device is "placed" after successfully making a vNic (it's possible to exit before this)
      for (int i = 0; i < vProps.ndevs; i++) {
        placedDevs[vProps.devs[i]] = 1;
      }
    } else {
      WARN("TOPO/NET : Could not force merge NICs %s. Please specify a valid NCCL_NET_FORCE_MERGE string.", semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    semi = strtok_r(NULL, ";", &semi_token);
  }

exit:
  free(ncStr);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTopoAutoMerge(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int* placedDevs,
                               ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs) {
  // Compute the path type between each device
  int* paths = NULL;
  ncclResult_t res = ncclSuccess;
  ncclCalloc(&paths, nPhysDevs * nPhysDevs);
  TRACE(NCCL_GRAPH, "Allocated %d paths", nPhysDevs * nPhysDevs);
  for (int i = 0; i < nPhysDevs; i++) {
    for (int j = 0; j < nPhysDevs; j++) {
      struct ncclXmlNode* nodes[2];
      nodes[0] = physNetNodes[i];
      nodes[1] = physNetNodes[j];
      struct ncclXmlNode* parent;
      NCCLCHECKGOTO(ncclTopoGetPath(nodes, 2, &paths[i * nPhysDevs + j], &parent), res, out);
    }
  }

  // Place all remaining physical devices into a virtual device given the mergeLevel criteria
  for (int i = 0; i < nPhysDevs; i++) {
    // Select the first unplaced device "i" as the root
    if (placedDevs[i] == 0) {
      // Init a new vDevice
      ncclNetVDeviceProps_t vProps;
      vProps = {0};
      vProps.devs[vProps.ndevs++] = i;
      placedDevs[i] = 1;
      TRACE(NCCL_GRAPH, "Placed dev %d", i);

      // Select each unplaced device "j" which is at most "mergeLevel" distance from "i", but not equal to "i"
      // (Don't merge the same device with itself)
      for (int j = 0; j < nPhysDevs; j++) {
        if ((paths[i * nPhysDevs + j] <= netInfo->mergeLevel) && (placedDevs[j] == 0 && j != i) &&
            (netInfo->mergePolicy != NCCL_NET_MERGE_POLICY_RAIL ||
             (propsList[i].railId != NCCL_NET_ID_UNDEF && propsList[i].railId == propsList[j].railId))) {
          vProps.devs[vProps.ndevs++] = j;
          placedDevs[j] = 1;
          TRACE(NCCL_GRAPH, "Placed dev %d path=%d", j, paths[i * nPhysDevs + j]);
        }
        if (vProps.ndevs == netInfo->maxDevsPerNic) break;
      }

      if (vProps.ndevs > netInfo->maxDevsPerNic) {
        WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps.ndevs, netInfo->maxDevsPerNic);
        return ncclInternalError;
      }

      ncclResult_t ret = ncclTopoMakeVnic(xml, netInfo, &vProps, physNetNodes);

      // Merging failed.
      // Mark all as unplaced and increase their distance to disconnected (PATH_DIS)
      // Set i to 0 to restart the automatic merging process and ensure all are placed
      if (ret != ncclSuccess) {
        INFO(NCCL_GRAPH | NCCL_INIT | NCCL_NET,
             "Marking physical devices as unplaced, increasing distance and restarting search.");
        placedDevs[i] = 0;
        TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, keeping distance -> self as PATH_LOC", i);
        for (int k = 1; k < vProps.ndevs; k++) {
          int dev = vProps.devs[k];
          placedDevs[dev] = 0;
          paths[i * nPhysDevs + dev] = PATH_DIS;
          paths[dev * nPhysDevs + i] = PATH_DIS;
          TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, setting distance -> %d as PATH_DIS", dev, i);
        }
        i = 0;
      }
    }
  }

out:
  free(paths);
  return res;
}

struct kvDict nicPathKvList[] = {{"LOC", PATH_LOC}, {"PORT", PATH_PORT}, {"PIX", PATH_PIX},
                                 {"PXB", PATH_PXB}, {"P2C", PATH_P2C},   {"PXN", PATH_PXN},
                                 {"PHB", PATH_PHB}, {"SYS", PATH_SYS},   {NULL, 0}};

ncclResult_t ncclTopoFindLinkWidthRec(ncclXmlNode* node, ncclXmlNode** physNetNodes, int ndevs, int* foundPhysNet,
                                      int* linkWidth) {
  int myLinkWidth = 0;
  if (strcmp(node->name, "pci") == 0) {
    NCCLCHECK(xmlGetAttrInt(node, "link_width", &myLinkWidth));
#ifdef ENABLE_TRACE
    const char *busidAttr, *linkAttr;
    NCCLCHECK(xmlGetAttrStr(node, "busid", &busidAttr));
    NCCLCHECK(xmlGetAttr(node, "link_width", &linkAttr));
    TRACE(NCCL_GRAPH, "Found link_width (%s)=%d for busid=%s", linkAttr, myLinkWidth, busidAttr);
#endif
  }

  *foundPhysNet = 0;
  // Detect if a physical child is found. This information will be propagated up the stack.
  int devId = 0;
  while (devId < ndevs && !(*foundPhysNet)) *foundPhysNet = (node == physNetNodes[devId++]);

  int totalChildLinkWidth = 0;
  for (int i = 0; i < node->nSubs; i++) {
    ncclXmlNode* child = node->subs[i];
    int found = 0;
    int tempLinkWidth = 0;
    NCCLCHECK(ncclTopoFindLinkWidthRec(child, physNetNodes, ndevs, &found, &tempLinkWidth));
    if (found) {
      *foundPhysNet = 1;
      totalChildLinkWidth += tempLinkWidth;
    }
  }

  if (*foundPhysNet == 0) {
    // No child NICs were found, do not accrue any detected link_width
    *linkWidth = 0;
    INFO(NCCL_GRAPH, "Did not find child net device. Returning link_width=%d totalChildLinkWidth=%d", *linkWidth,
         totalChildLinkWidth);
  } else if (totalChildLinkWidth == 0) {
    // If A child NIC was found but no link_width was detected among children, assign the link_width to mine (I am
    // the first pci node right above the physNetNode).
    *linkWidth = myLinkWidth;
    INFO(NCCL_GRAPH, "Found child net device for %s. Returning link_width=%d totalChildLinkWidth=%d", node->name,
         *linkWidth, totalChildLinkWidth);
  } else {
    // Standard recursive accrual of link_width. The link_width is either the bottleneck of this PCI node's width or the sum of its children's width.
    *linkWidth = myLinkWidth > 0 ? std::min(myLinkWidth, totalChildLinkWidth) : totalChildLinkWidth;
    INFO(NCCL_GRAPH, "Found child net device for %s. Returning link_width=%d totalChildLinkWidth=%d", node->name,
         *linkWidth, totalChildLinkWidth);
  }

  return ncclSuccess;
}

// DFS over nodes under common parent
// Exclude link widths of non-physNetNodes chains
ncclResult_t ncclTopoFindLinkWidth(ncclXmlNode* parent, ncclXmlNode** physNetNodes, int ndevs, int* linkWidth) {
  *linkWidth = 0;
  for (int i = 0; i < parent->nSubs; i++) {
    ncclXmlNode* child = parent->subs[i];
    int foundPhysNet = 0;
    int childLinkWidth = 0;
    NCCLCHECK(ncclTopoFindLinkWidthRec(child, physNetNodes, ndevs, &foundPhysNet, &childLinkWidth));
    if (foundPhysNet) {
      *linkWidth += childLinkWidth;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoGetVNicParent(struct ncclXml* xml, const char* tagName,
                                   ncclResult_t (*getProperties)(int, ncclNetProperties_t*),
                                   ncclNetVDeviceProps_t* vProps, ncclXmlNode** parent) {
  ncclNetProperties_t props[NCCL_NET_MAX_DEVS_PER_NIC];
  ncclXmlNode* physNetNodes[NCCL_NET_MAX_DEVS_PER_NIC];
  for (int i = 0; i < vProps->ndevs; i++) {
    NCCLCHECK(getProperties(vProps->devs[i], props + i));
    struct ncclXmlNode* physNetNode;
    NCCLCHECK(xmlFindTagKv(xml, tagName, &physNetNode, "name", props[i].name));
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Re-found physical ncclNet node %d %s", i, props[i].name);
  }

  int path = PATH_LOC;
  NCCLCHECK(ncclTopoGetPath(physNetNodes, vProps->ndevs, &path, parent));
  int aggregateWidth = 0;
  if (path == PATH_PHB || path == PATH_PXB || path == PATH_PIX) {
    NCCLCHECK(ncclTopoFindLinkWidth(*parent, physNetNodes, vProps->ndevs, &aggregateWidth));
  }

  // If the common parent is a PCI switch or the CPU, we must reparent the new NIC under a made up pci device with a
  // unique busid
  // This adds a pci node between the physNetParent and the fused NIC.
  struct ncclXmlNode* physNetParent = *parent;
  if (*parent) {
    if (strcmp((*parent)->name, "pci") == 0) {
      // Compare PCI class here to avoid NCCL WARN when the "class" attribute doesn't exist
      const char* c;
      NCCLCHECK(xmlGetAttrStr(*parent, "class", &c));
      if (c && strcmp(c, PCI_BRIDGE_DEVICE_CLASS) == 0) {
        NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
      }
    } else if (strcmp((*parent)->name, "cpu") == 0) {
      // If the common parent is a CPU node, we must reparent the new NIC under a made up pci device with a unique busid
      NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
    } else if (strcmp((*parent)->name, "system") == 0) {
      // Cross-NUMA merge: common parent is the system root, which ncclTopoGetSystemFromXml
      // does not traverse for NICs (only iterates <cpu> children). Reparent under the first
      // physical NIC's CPU so the vNIC appears inside a proper <cpu> subtree.
      *parent = physNetNodes[0]->parent; // <nic> node
      while (*parent && strcmp((*parent)->name, "cpu") != 0) *parent = (*parent)->parent;
      if (*parent == NULL) {
        WARN("TOPO/NET : Cannot find CPU parent for cross-NUMA vNIC, attaching to first CPU");
        NCCLCHECK(xmlFindTag(xml, "cpu", parent));
        if (*parent == NULL) {
          WARN("TOPO/NET : No CPU node found in topology, cannot place cross-NUMA vNIC");
          return ncclInternalError;
        }
      }
      NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
    } else if (strcmp((*parent)->name, "system") == 0) {
      WARN("Fusing NET devices from different NUMA domains is not supported.");
      return ncclInvalidArgument;
    }
  }

  // Update the speed for all the pci nodes between the parent and the physNetparent
  if (aggregateWidth > 0) {
    struct ncclXmlNode* node = *parent;
    while (node) {
      TRACE(NCCL_GRAPH, "Set link_width to %d for vNIC parent %s", aggregateWidth, (*parent)->name);
      if (strcmp(node->name, "pci") == 0) NCCLCHECK(xmlSetAttrInt(node, "link_width", aggregateWidth));
      if (node == physNetParent) break;
      node = node->parent;
    }
  }

  TRACE(NCCL_GRAPH, "Selected parent %s with path %d", (*parent)->name, path);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeVNics(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int physicalDevs) {
  int* placedDevs = NULL;
  struct ncclXmlNode** physNetNodes = NULL;
  ncclNetProperties_t* props = NULL;
  ncclResult_t res = ncclSuccess;
  if (physicalDevs == 0) return ncclSuccess;

  NCCLCHECKGOTO(ncclCalloc(&physNetNodes, physicalDevs), res, out);
  NCCLCHECKGOTO(ncclCalloc(&placedDevs, physicalDevs), res, out);
  NCCLCHECKGOTO(ncclCalloc(&props, physicalDevs), res, out);
  for (int i = 0; i < physicalDevs; i++) {
    NCCLCHECKGOTO(netInfo->getProperties(i, props + i), res, out);
    struct ncclXmlNode* physNetNode;
    NCCLCHECKGOTO(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name), res, out);
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Found physical ncclNet node %d %s", i, props[i].name);
  }

  if (netInfo->forceMerge)
    NCCLCHECKGOTO(ncclTopoForceMerge(xml, netInfo, placedDevs, props, physNetNodes, physicalDevs), res, out);
  NCCLCHECKGOTO(ncclTopoAutoMerge(xml, netInfo, placedDevs, props, physNetNodes, physicalDevs), res, out);

out:
  free(physNetNodes);
  free(props);
  free(placedDevs);
  return res;
}

static ncclResult_t ncclTopoPopulateNics(ncclXml* xml, int startIndex, int endIndex, struct ncclTopoNetInfo* netInfo,
                                         int virtualNics) {
  const char* tagName = netInfo->gin ? "gin" : (netInfo->rma ? "rma" : "net");
  for (int n = startIndex; n < endIndex; n++) {
    ncclNetProperties_t props;
    NCCLCHECK(netInfo->getProperties(n, &props));
    struct ncclXmlNode* netNode = NULL;
    struct ncclXmlNode* parent = NULL;
    if (virtualNics) {
      struct ncclXmlNode* net = NULL;
      NCCLCHECK(xmlFindTagKv(xml, tagName, &net, "name", props.name));
      // In the event of multithreaded use case, we need to re-discover the shared parent of the given devices for
      // this vNIC
      // Only run this if the net doesn't exist locally - this may alter the XML state
      if (net == NULL) NCCLCHECK(ncclTopoGetVNicParent(xml, tagName, netInfo->getProperties, &props.vProps, &parent));
    }

    NCCLCHECK(ncclTopoFillNet(xml, tagName, props.pciPath, props.name, &netNode, parent));
    // Move <net .... gin/rma=1> XML nodes to <gin/rma ...> nodes.
    // This would lead to duplication of the device in ncclTopoAddNic.
    if (netInfo->gin || netInfo->rma) {
      struct ncclXmlNode* legacyNetNode = NULL;
      NCCLCHECK(xmlFindTagKv(xml, "net", &legacyNetNode, "name", props.name));
      if (legacyNetNode) {
        int legacyAttr = 0;
        NCCLCHECK(xmlGetAttrIntDefault(legacyNetNode, tagName, &legacyAttr, 0));
        if (legacyAttr) {
          int count = 0;
          while (count < legacyNetNode->nAttrs) {
            NCCLCHECK(xmlSetAttrIfUnset(netNode, legacyNetNode->attrs[count].key, legacyNetNode->attrs[count].value));
            count++;
          }
          NCCLCHECK(xmlUnsetAttr(legacyNetNode, tagName));
        }
      }
    }

    const char* colAttr;
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));

    NCCLCHECK(xmlSetAttrInt(netNode, "keep", 1));
    int dev;
    xmlGetAttrIntDefault(netNode, "dev", &dev, -1);
    if (dev != -1 && dev != n) {
      int net = 0, gin = 0, rma = 0;
      xmlGetAttrIntDefault(netNode, "net", &net, dev >= 0); // no "net" attribute uses the presence of dev
      xmlGetAttrIntDefault(netNode, "gin", &gin, 0);
      xmlGetAttrIntDefault(netNode, "rma", &rma, 0);
      ATTN("TOPO/NET : Changing %s dev index from %d to %d. This device is shared by %s %s %s plugin(s). This can be "
           "due to outdated plugin or topology xml file. The former may cause performance degradation. Try setting "
           "NCCL_IB_DEVICE_PCI_ORDER=0 or upgrading your net plugin to resolve this warning.",
           netInfo->name, dev, n, net ? "NET" : "", gin ? "GIN" : "", rma ? "RMA" : "");
    }

    NCCLCHECK(xmlSetAttrInt(netNode, "dev", n));
    NCCLCHECK(xmlInitAttrInt(netNode, "latency", props.latency));
    NCCLCHECK(xmlInitAttrInt(netNode, "speed", props.speed));
    NCCLCHECK(xmlInitAttrInt(netNode, "port", props.port));
    // Overwrite stale GUID from topo injection if it exists
    NCCLCHECK(xmlSetAttrUint64(netNode, "guid", props.guid));
    // Reserve bit 16 for NCCL-generated rail and plane IDs.
    int rail = (props.railId == NCCL_NET_ID_UNDEF) ? NCCL_TOPO_UNDEF : (props.railId & 0xFFFF);
    int plane = (props.planeId == NCCL_NET_ID_UNDEF) ? NCCL_TOPO_UNDEF : (props.planeId & 0xFFFF);
    NCCLCHECK(ncclTopoGetRailPlane(netInfo->railKeyList, netNode, &rail, &plane));

    int oldRail, oldPlane;
    NCCLCHECK(xmlGetAttrIntDefault(netNode, "rail", &oldRail, rail));
    NCCLCHECK(xmlGetAttrIntDefault(netNode, "plane", &oldPlane, plane));
    // [RCCL] Tolerate a mismatch between two NCCL-generated values (bit 16 set on both): topology XML dumped by
    // RCCL before 2.32.6 carries plane = UNDEF_BIT | port, where NCCL now numbers same-tag siblings instead. Take
    // the newly generated value. A plugin-provided value on either side must still match.
    bool railGenerated = (oldRail & NCCL_TOPO_UNDEF_BIT) && (rail & NCCL_TOPO_UNDEF_BIT);
    bool planeGenerated = (oldPlane & NCCL_TOPO_UNDEF_BIT) && (plane & NCCL_TOPO_UNDEF_BIT);
    if ((oldRail != rail && !railGenerated) || (oldPlane != plane && !planeGenerated)) {
      WARN("TOPO/NET : Inconsistent rail/plane for net %s: existing %d/%d, new %d/%d.", props.name, oldRail, oldPlane,
           rail, plane);
      return ncclInvalidUsage;
    }
    if (oldRail != rail || oldPlane != plane) {
      INFO(NCCL_GRAPH | NCCL_NET, "TOPO/NET : Regenerating rail/plane for net %s: existing %d/%d, new %d/%d.",
           props.name, oldRail, oldPlane, rail, plane);
    }
    NCCLCHECK(xmlSetAttrInt(netNode, "rail", rail));
    NCCLCHECK(xmlSetAttrInt(netNode, "plane", plane));

    NCCLCHECK(xmlInitAttrInt(netNode, "maxconn", props.maxComms));
    bool gdrSupport =
      (props.ptrSupport & NCCL_PTR_CUDA) || (netInfo->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF));
    INFO(NCCL_NET, "NET/%s : GPU Direct RDMA %s for HCA %d '%s'", netInfo->name, gdrSupport ? "Enabled" : "Disabled", n,
         props.name);
    NCCLCHECK(xmlInitAttrInt(netNode, "gdr", gdrSupport));

    // GIN and COLL plugins must set "net=0"; it's absence is interpretted as "net=1" in ncclTopoAddNic.
    int isNet = 0;
    const char* netAttr = NULL;
    NCCLCHECK(xmlGetAttr(netNode, "net", &netAttr));
    if (netAttr) isNet = strtol(netAttr, NULL, 0);
    NCCLCHECK(xmlSetAttrInt(netNode, "net", netInfo->net || isNet));
    // Only set coll or gin if it's not 0
    if (netInfo->coll) NCCLCHECK(xmlInitAttrInt(netNode, "coll", netInfo->coll));
    if (netInfo->gin) NCCLCHECK(xmlInitAttrInt(netNode, "gin", netInfo->gin));
    if (netInfo->rma) NCCLCHECK(xmlInitAttrInt(netNode, "rma", netInfo->rma));

    const char *keepAttr, *ginAttr, *rmaAttr;
    NCCLCHECK(xmlGetAttr(netNode, "net", &netAttr));
    NCCLCHECK(xmlGetAttr(netNode, "gin", &ginAttr));
    NCCLCHECK(xmlGetAttr(netNode, "rma", &rmaAttr));
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));
    NCCLCHECK(xmlGetAttr(netNode, "keep", &keepAttr));
    INFO(NCCL_GRAPH,
         "ncclTopoPopulateNics : Filled %s in topo with pciPath=%s net=%s gin=%s rma=%s keep=%s coll=%s rail=%d (%d) "
         "plane=%d (%d)",
         props.name, props.pciPath, netAttr, ginAttr, rmaAttr, keepAttr, colAttr, rail, props.railId, plane,
         props.planeId);
  }

  return ncclSuccess;
}

static ncclResult_t ncclTopoUpdateVNics(ncclXml* xml, struct ncclTopoNetInfo* net, int nPhysicalNics,
                                        int nVirtualNics) {
  for (int n = nPhysicalNics; n < nPhysicalNics + nVirtualNics; n++) {
    ncclNetProperties_t vProps;
    NCCLCHECK(net->getProperties(n, &vProps));
    for (int i = 0; i < vProps.vProps.ndevs; i++) {
      ncclNetProperties_t physProps;
      NCCLCHECK(net->getProperties(vProps.vProps.devs[i], &physProps));
      struct ncclXmlNode* physNetNode = NULL;
      NCCLCHECK(xmlFindTagKv(xml, "net", &physNetNode, "name", physProps.name));
      if (physNetNode) {
        NCCLCHECK(xmlSetAttrInt(physNetNode, net->net ? "net" : (net->gin ? "gin" : "coll"), 0));
        // net is always present (see ncclTopoPopulateNics).
        int net = 0, gin = 0, coll = 0;
        NCCLCHECK(xmlGetAttrInt(physNetNode, "net", &net));
        NCCLCHECK(xmlGetAttrIntDefault(physNetNode, "gin", &gin, 0));
        NCCLCHECK(xmlGetAttrIntDefault(physNetNode, "coll", &coll, 0));
        // Set "keep = 0" only if no plugin is using the physical device
        if (net == 0 && gin == 0 && coll == 0) NCCLCHECK(xmlSetAttrInt(physNetNode, "keep", 0));
      }
    }
  }
  return ncclSuccess;
}

// Calls to network plugin APIs should be protected. This function should be called inside a per-process lock.
ncclResult_t ncclTopoProcessNet(ncclXml* xml, const char* dumpXmlFile, struct ncclTopoNetInfo* net) {
  bool usePhysicalDevices = (dumpXmlFile || net->makeVDevice == NULL);
  int nPhysicalNics, nVirtualNics;
  NCCLCHECK(net->getDevCount(net->netPluginIndex, &nPhysicalNics, &nVirtualNics));
  // List the physical devices in the topo and set keep = 1
  NCCLCHECK(ncclTopoPopulateNics(xml, 0, nPhysicalNics, net, /*virtual=*/false));
  if (!usePhysicalDevices) {
    // Virtual devices are only created once per network
    if (nVirtualNics == NCCL_UNDEF_DEV_COUNT) {
      NCCLCHECK(ncclTopoMakeVNics(xml, net, nPhysicalNics));
      // Update the number of virtual devices both locally and in the state tracking the plugin.
      // Note: 0 is a valid number of virtual devices
      int nDevs;
      NCCLCHECK(net->devices(&nDevs));
      nVirtualNics = nDevs - nPhysicalNics;
      NCCLCHECK(net->setVirtDevCount(net->netPluginIndex, nVirtualNics));
    }
    // populate the virtual devices if any
    if (nVirtualNics > 0) {
      // Physical devices within a virtual one have to be removed. The fused dev inherits the guid.
      // Note: ncclTopoMakeVnic doesn't create a vNic if ndevs = 1; no special case needed
      NCCLCHECK(ncclTopoUpdateVNics(xml, net, nPhysicalNics, nVirtualNics));
      // Populate the virtual devices and set keep = 1
      NCCLCHECK(ncclTopoPopulateNics(xml, nPhysicalNics, nPhysicalNics + nVirtualNics, net, /*virtual=*/true));
    }
  }

  return ncclSuccess;
}

static bool ncclTopoIsVeraRubin(int cudaCompCap, int cpuArch) {
  return cpuArch == NCCL_TOPO_CPU_ARCH_ARM && RUBIN_AND_LATER(cudaCompCap);
}

ncclResult_t ncclTopoGetFusionEnv(int* mergeLevel, const char** forceMerge, int cudaCompCap, int cpuArch) {
  if (forceMerge) *forceMerge = ncclGetEnv("NCCL_NET_FORCE_MERGE");
  const char* mergeLevelEnv = ncclGetEnv("NCCL_NET_MERGE_LEVEL");
  if (mergeLevelEnv) {
    kvConvertToInt(mergeLevelEnv, mergeLevel, nicPathKvList);
  } else {
    *mergeLevel = ncclTopoIsVeraRubin(cudaCompCap, cpuArch) ? PATH_PHB : PATH_PORT;
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetMergePolicy(int* mergePolicy, int cudaCompCap, int cpuArch) {
  *mergePolicy = ncclTopoIsVeraRubin(cudaCompCap, cpuArch) ? NCCL_NET_MERGE_POLICY_RAIL : NCCL_NET_MERGE_POLICY_ALL;
  const char* env = ncclGetEnv("NCCL_NET_MERGE_POLICY");
  if (env) {
    if (strcasecmp(env, "RAIL") == 0) {
      *mergePolicy = NCCL_NET_MERGE_POLICY_RAIL;
      INFO(NCCL_ENV, "NCCL_NET_MERGE_POLICY set by environment to RAIL");
    } else {
      *mergePolicy = NCCL_NET_MERGE_POLICY_ALL;
      if (strcasecmp(env, "ALL") != 0) {
        WARN("NCCL_NET_MERGE_POLICY: unknown value '%s', defaulting to ALL", env);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoImportNets(struct ncclXml* xml, const struct ncclTopoDiscoveryConfig* config, int cpuArch) {
  std::unique_lock<std::mutex> lock;
  if (config->netLock) lock = std::unique_lock<std::mutex>(*config->netLock);
  INFO(NCCL_GRAPH, "TOPO/NET : Importing network plugins to topology");
  struct ncclTopoNetRailKeyList railKeyList;
  for (int i = 0; i < config->nNets; i++) {
    struct ncclTopoNetInfo netInfo = config->nets[i];
    netInfo.railKeyList = &railKeyList;
    if (netInfo.coll || netInfo.net) {
      NCCLCHECK(ncclTopoGetFusionEnv(&netInfo.mergeLevel, &netInfo.forceMerge, config->minCompCap, cpuArch));
      NCCLCHECK(ncclTopoGetMergePolicy(&netInfo.mergePolicy, config->minCompCap, cpuArch));
    }
    NCCLCHECK(ncclTopoProcessNet(xml, config->dumpXmlFile, &netInfo));
  }
  return ncclSuccess;
}
