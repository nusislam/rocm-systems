/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/graph/topo.cc for every binary that needs it, whether or not it is the
// code under test. It is #included rather than linked because
// TopoSetNetDevsPolicy can only be defined inside topo.cc's TU, where the
// NETDEVS policy statics live. topo-test.cc #includes this file for the same
// reason, so the definition exists exactly once. The NIC import and XML
// fusion units of src/graph/discovery/ come along so topo-test.cc can reach
// their statics; device discovery (topo_discovery_dev.cc) is linked.

#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include TOPO_CC_PATH
#include TOPO_DISCOVERY_NET_CC_PATH
#include TOPO_DISCOVERY_CC_PATH

#include "TopoScene.h"

// The NETDEVS policy lives in this TU's statics; the other graph TUs set it through here.
void TopoSetNetDevsPolicy(enum netDevsPolicy policy, int num) {
  int unused;
  ncclTopoGetNetDevsPolicy(nullptr, &unused);  // consume the call_once first
  netDevsPolicy = policy;
  netDevsPolicyNum = num;
}
