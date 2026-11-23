/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fused XML construction: builds this process's topology XML, then fuses it with the XML of every rank sharing
// the node.

#include "core.h"
#include "topo_discovery.h"
#include "../topo.h"
#include <cstring>

ncclResult_t ncclTopoGetLocalXml(const struct ncclTopoDiscoveryConfig* config, struct ncclXml** xmlOut) {
  ncclResult_t ret = ncclSuccess;
  struct ncclXml* xml = NULL;
  struct ncclXmlNode* node;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  int cpuArch = NCCL_TOPO_UNDEF;

  NCCLCHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));
  if (config->xmlTopoFile) {
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile(config->xmlTopoFile, xml, 1), ret, fail);
  } else {
    // Try default XML topology location
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile("/var/run/nvidia-topologyd/virtualTopology.xml", xml, 0), ret, fail);
  }
  // Update every cpu node's host_hash attribute since those are not
  // intended to be preserved from the XML files that have been read.
  NCCLCHECKGOTO(xmlFindTag(xml, "cpu", &node), ret, fail);
  while (node != nullptr) {
    NCCLCHECKGOTO(xmlSetAttrLong(node, "host_hash", getHostHash()), ret, fail);
    NCCLCHECKGOTO(xmlFindNextTag(xml, "cpu", node, &node), ret, fail);
  }
  if (xml->maxIndex == 0) {
    // Create top tag
    struct ncclXmlNode* top;
    NCCLCHECKGOTO(xmlAddNode(xml, NULL, "system", &top), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION), ret, fail);
  }

  NCCLCHECKGOTO(ncclTopoRefreshBcmP2pLinks(), ret, fail);

  // Detect only the GPU managed by this process.  We'll get any others through XML fusion.
  NCCLCHECKGOTO(int64ToBusId(config->busId, busId), ret, fail);
  NCCLCHECKGOTO(ncclTopoFillGpu(xml, busId, &node), ret, fail);
  if (node) {
    NCCLCHECKGOTO(xmlSetAttrInt(node, "keep", 1), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(node, "rank", config->rank), ret, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(node, "gdr", config->gdrSupport), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(node, "mlopart", config->mloPart), ret, fail);
  }
  NCCLCHECKGOTO(ncclTopoGetXmlCpuArch(xml, &cpuArch), ret, fail);

  // Auto-detect NICs if needed, net/gin/collnet share the same xml/graph nodes.
  NCCLCHECKGOTO(ncclTopoImportNets(xml, config, cpuArch), ret, fail);

  // Remove XML branches which don't have a node with keep="1" (typically when importing a topology)
  NCCLCHECKGOTO(ncclTopoTrimXml(xml), ret, fail);

  *xmlOut = xml;
  return ncclSuccess;
fail:
  free(xml);
  return ret;
}

ncclResult_t ncclTopoFuseLocalXml(const struct ncclTopoDiscoveryConfig* config, struct ncclXml** xml) {
  ncclResult_t ret = ncclSuccess;
  // Every rank contributes a full-size XML so that all of them agree on the slot size.
  const size_t xmlSize = xmlMemSize(NCCL_TOPO_XML_MAX_NODES);
  char* mem = NULL;
  struct ncclXml* rankXml;
  int fusedMaxNodes = 0;

  if (config->nLocalRanks <= 0 || config->localRank < 0 || config->localRank >= config->nLocalRanks ||
      (config->allGather == NULL && config->nLocalRanks > 1)) {
    WARN("ncclTopoFuseLocalXml: cannot fuse as local rank %d of %d", config->localRank, config->nLocalRanks);
    return ncclInvalidArgument;
  }
  if ((*xml)->maxNodes != NCCL_TOPO_XML_MAX_NODES) {
    WARN("ncclTopoFuseLocalXml: XML holds %d nodes, %d expected", (*xml)->maxNodes, NCCL_TOPO_XML_MAX_NODES);
    return ncclInternalError;
  }

  NCCLCHECKGOTO(ncclCalloc(&mem, config->nLocalRanks * xmlSize), ret, exit);
  rankXml = (struct ncclXml*)(mem + xmlSize * config->localRank);
  memcpy(rankXml, *xml, xmlSize);
  NCCLCHECKGOTO(ncclTopoConvertXml(rankXml, (uintptr_t)(*xml)->nodes, 1), ret, exit);
  if (config->allGather) NCCLCHECKGOTO(config->allGather(config->allGatherCtx, mem, xmlSize), ret, exit);
  // ncclTopoFuseXml only copies existing source nodes into dst (never adds new
  // ones), so fusedMaxNodes is a true upper bound on the fused node count.
  for (int i = 0; i < config->nLocalRanks; i++) {
    struct ncclXml* peerXml = (struct ncclXml*)(mem + xmlSize * i);
    fusedMaxNodes += peerXml->maxIndex;
  }
  if (fusedMaxNodes > (*xml)->maxNodes) {
    free(*xml);
    *xml = NULL;
    NCCLCHECKGOTO(xmlAlloc(xml, fusedMaxNodes), ret, exit);
  } else {
    // The original topology is preserved in mem, so reuse this allocation as the empty fusion destination.
    (*xml)->maxIndex = 0;
  }
  for (int i = 0; i < config->nLocalRanks; i++) {
    struct ncclXml* peerXml = (struct ncclXml*)(mem + xmlSize * i);
    NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, exit);
    NCCLCHECKGOTO(ncclTopoFuseXml(*xml, peerXml), ret, exit);
  }

exit:
  free(mem);
  return ret;
}

ncclResult_t ncclTopoDiscoverXml(const struct ncclTopoDiscoveryConfig* config, struct ncclXml** xml) {
  ncclResult_t ret = ncclSuccess;
  struct ncclXmlNode* topNode;
  *xml = NULL;
  NCCLCHECKGOTO(ncclTopoGetLocalXml(config, xml), ret, fail);
  NCCLCHECKGOTO(ncclTopoFuseLocalXml(config, xml), ret, fail);
  NCCLCHECKGOTO(xmlFindTag(*xml, "system", &topNode), ret, fail);
  if (topNode) NCCLCHECKGOTO(xmlSetAttrInt(topNode, "cross_clique", config->crossClique), ret, fail);
  return ncclSuccess;
fail:
  free(*xml);
  *xml = NULL;
  return ret;
}
