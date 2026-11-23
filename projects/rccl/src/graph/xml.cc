/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <ctype.h>
#include <float.h>
#include "core.h"
#include "xml.h"
#include "archinfo.h"
#include "os.h"
#include "topo.h"

// Arbitrarily large number for constructing virtual topology string
#define NCCL_MAX_XML_DEPTH 1024

/*******************/
/* XML File Parser */
/*******************/

typedef ncclResult_t (*xmlHandlerFunc_t)(FILE*, struct ncclXml*, struct ncclXmlNode*);

struct xmlHandler {
  const char* name;
  xmlHandlerFunc_t func;
};

#if defined(NCCL_OS_LINUX)

ncclResult_t xmlGetChar(FILE* file, char* c) {
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  char c;
  NCCLCHECK(xmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o] = c;
      if (o == MAX_STR_LEN - 1) {
        value[o] = '\0';
        WARN("Error : value %s too long (max %d)", value, MAX_STR_LEN);
        return ncclInternalError;
      }
      o++;
      NCCLCHECK(xmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return ncclSuccess;
#else
    WARN("XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  int o = 0;
  char quote = c;  // Remember which quote type we started with
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    value[o] = c;
    if (o == MAX_STR_LEN - 1) {
      value[o] = '\0';
      WARN("Error : value %s too long (max %d)", value, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != quote);
  value[o - 1] = '\0';
  NCCLCHECK(xmlGetChar(file, last));
  return ncclSuccess;
}

ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return xmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN - 1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o - 1] = '\0';
  *last = c;
  return ncclSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) \
  do { \
    s[0] = s[1]; \
    s[1] = s[2]; \
    s[2] = c; \
  } while (0)
ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i = 0; i < strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    NCCLCHECK(xmlSkipComment(file, node->name + 3, c));
    return xmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)", MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(xmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[],
                        int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (xml->maxIndex == xml->maxNodes) {
      WARN("Error : XML parser is limited to %d nodes", xml->maxNodes);
      return ncclInternalError;
    }
    struct ncclXmlNode* node = xml->nodes + xml->maxIndex;
    memset(node, 0, sizeof(struct ncclXmlNode));
    NCCLCHECK(xmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h = 0; h < nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) {
          if (head->nSubs == MAX_SUBS) {
            WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
            return ncclInternalError;
          }
          head->subs[head->nSubs++] = node;
        }
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(xmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

#endif

/**************/
/* XML Writer */
/**************/

// exp == 1 -- serialize; exp == 0 -- deserialize
ncclResult_t ncclTopoConvertXml(struct ncclXml* xml, uintptr_t base, int exp) {
  for (int n = 0; n < xml->maxIndex; n++) {
    struct ncclXmlNode* node = &xml->nodes[n];

    // For "parent", we shift the base by 1 so that we can distinguish actual
    // NULL pointers from pointers pointing to the first node.
    if (node->parent)
      node->parent =
        (struct ncclXmlNode*)(exp ? ((uintptr_t)node->parent - base + 1) : (base - 1 + (uintptr_t)node->parent));

    for (int s = 0; s < node->nSubs; s++) {
      node->subs[s] =
        (struct ncclXmlNode*)(exp ? ((uintptr_t)node->subs[s] - base) : (base + (uintptr_t)node->subs[s]));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlRec(int indent, FILE* file, struct ncclXmlNode* node) {
  for (int i = 0; i < indent; i++) fprintf(file, " ");
  fprintf(file, "<%s", node->name);

  for (int a = 0; a < node->nAttrs; a++) {
    // Skip NCCL-generated rail/plane IDs
    if (strcmp(node->attrs[a].key, "rail") == 0 || strcmp(node->attrs[a].key, "plane") == 0) {
      if (strtol(node->attrs[a].value, NULL, 0) & NCCL_TOPO_UNDEF_BIT) continue;
    }
    fprintf(file, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].value);
  }
  if (node->nSubs == 0) {
    fprintf(file, "/>\n");
  } else {
    fprintf(file, ">\n");
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoDumpXmlRec(indent + 2, file, node->subs[s]));
    }
    for (int i = 0; i < indent; i++) fprintf(file, " ");
    fprintf(file, "</%s>\n", node->name);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml) {
  ncclResult_t ret = ncclSuccess;
  FILE* file = fopen(xmlTopoFile, "w");
  if (file == NULL) {
    INFO(NCCL_GRAPH | NCCL_ENV, "Unable to open %s, not dumping topology.", xmlTopoFile);
    return ncclSuccess;
  }
  NCCLCHECKGOTO(ncclTopoDumpXmlRec(0, file, xml->nodes), ret, exit);
exit:
  fclose(file);
  return ret;
}

static ncclResult_t xmlTopoFuseXmlRecursive(struct ncclXml* dst, struct ncclXmlNode* dstParent,
                                            struct ncclXmlNode* srcParent) {
  for (int i = 0; i < srcParent->nSubs; i++) {
    struct ncclXmlNode* srcNode = srcParent->subs[i];
    struct ncclXmlNode* dstNode;
    NCCLCHECK(xmlFindNode(dstParent, srcNode, &dstNode));
    if (dstNode == NULL) {
      NCCLCHECK(xmlAddTree(dst, dstParent, srcNode));
    } else {
      NCCLCHECK(xmlTopoFuseXmlRecursive(dst, dstNode, srcNode));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src) {
  struct ncclXmlNode* topNodeDst;
  NCCLCHECK(xmlFindTag(dst, "system", &topNodeDst));

  if (topNodeDst == NULL) {
    NCCLCHECK(xmlAddTree(dst, NULL, src->nodes));
    return ncclSuccess;
  }

  struct ncclXmlNode* topNodeSrc;
  NCCLCHECK(xmlFindTag(src, "system", &topNodeSrc));

  NCCLCHECK(xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc));

  return ncclSuccess;
}

#if NCCL_OS_LINUX

/****************************************/
/* Parser rules for our specific format */
/****************************************/

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPciLink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadC2c(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}
ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  struct xmlHandler handlers[] = {{"xgmi", ncclTopoXmlLoadNvlink}};
#else
  struct xmlHandler handlers[] = {{"nvlink", ncclTopoXmlLoadNvlink}, {"c2c", ncclTopoXmlLoadC2c}};
#endif
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, sizeof(handlers) / sizeof(handlers[0])));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {
    {"net", ncclTopoXmlLoadNet}, {"gin", ncclTopoXmlLoadNet}, {"rma", ncclTopoXmlLoadNet}
  };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 3));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"pci", ncclTopoXmlLoadPci},
                                  {"gpu", ncclTopoXmlLoadGpu},
                                  {"nic", ncclTopoXmlLoadNic},
                                  {"pcilink", ncclTopoXmlLoadPciLink}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 4));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"pci", ncclTopoXmlLoadPci}, {"nic", ncclTopoXmlLoadNic}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_TOPO_XML_VERSION) {
    WARN("XML Topology has wrong version %d, %d needed", version, NCCL_TOPO_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) {
    INFO(NCCL_GRAPH, "Loading topology %s", name);
  } else {
    INFO(NCCL_GRAPH, "Loading unnamed topology");
  }

  struct xmlHandler handlers[] = {{"cpu", ncclTopoXmlLoadCpu}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

#endif // NCCL_OS_LINUX (XML file parsers)

ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml, int warn) {
#if NCCL_OS_LINUX
  ncclResult_t ret = ncclSuccess;
  FILE* file = fopen(xmlTopoFile, "r");
  if (file == NULL) {
    if (warn) {
      INFO(NCCL_GRAPH | NCCL_ENV, "Could not open XML topology file %s : %s", xmlTopoFile, strerror(errno));
    }
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "Loading topology file %s", xmlTopoFile);
  struct xmlHandler handlers[] = {{"system", ncclTopoXmlLoadSystem}};
  xml->maxIndex = 0;
  NCCLCHECKGOTO(xmlLoadSub(file, xml, NULL, handlers, 1), ret, exit);
exit:
  fclose(file);
  return ret;
#elif NCCL_OS_WINDOWS
  (void)xmlTopoFile;
  (void)xml;
  (void)warn;
  return ncclSuccess;
#endif
}

ncclResult_t xmlUnsetAttr(struct ncclXmlNode* node, const char* attrName) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) return ncclSuccess;
  for (int i = index + 1; i < node->nAttrs; i++) {
    strcpy(node->attrs[i - 1].key, node->attrs[i].key);
    strcpy(node->attrs[i - 1].value, node->attrs[i].value);
  }
  node->nAttrs--;
  return ncclSuccess;
}

ncclResult_t ncclTopoTrimXmlRec(struct ncclXmlNode* node, int* keep) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, "keep", &str));
  if (str && strcmp(str, "1") == 0) {
    NCCLCHECK(xmlUnsetAttr(node, "keep"));
    *keep = 1;
  } else {
    // Copy nSubs and subs as they could change as we trim recursively.
    struct ncclXmlNode** subs = NULL;
    NCCLCHECK(ncclCalloc(&subs, MAX_SUBS));
    int nSubs = node->nSubs;
    memcpy(subs, node->subs, node->nSubs * sizeof(struct ncclXmlNode*));
    *keep = 0;
    ncclResult_t subsRes = ncclSuccess;
    for (int s = 0; s < nSubs; s++) {
      int k = 0;
      subsRes = ncclTopoTrimXmlRec(subs[s], &k);
      if (subsRes != ncclSuccess) {
        break;
      }
      *keep += k;
    }
    free(subs);
    NCCLCHECK(subsRes);
    // Remove node if it has no children and no keep attribute
    if (*keep == 0 && // Trim PCI switches, CPUs with no used GPU/NIC under them, or pruned NICs
        (strcmp(node->name, "pci") == 0 || strcmp(node->name, "cpu") == 0 || strcmp(node->name, "nic") == 0 ||
         strcmp(node->name, "net") == 0 || strcmp(node->name, "gin") == 0 || strcmp(node->name, "rma") == 0)) {
#ifdef ENABLE_TRACE
      const char* name;
      const char* busid;
      NCCLCHECK(xmlGetAttr(node, "name", &name));
      NCCLCHECK(xmlGetAttr(node, "busid", &busid));
      TRACE(NCCL_GRAPH, "Removing node %s %s %s\n", node->name, name, busid);
#endif
      NCCLCHECK(xmlRemoveNode(node));
    }
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoTrimXml(struct ncclXml* xml) {
  int keep = 0;
  NCCLCHECK(ncclTopoTrimXmlRec(xml->nodes, &keep));
  return ncclSuccess;
}

#if NCCL_OS_LINUX

/**************************************************/
/* Parser rules for the user-defined graph search */
/**************************************************/

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"net", ncclTopoXmlGraphLoadNet}, {"gpu", ncclTopoXmlGraphLoadGpu}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"channel", ncclTopoXmlGraphLoadChannel}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_GRAPH_XML_VERSION) {
    WARN("XML Graph has wrong version %d, %d needed", version, NCCL_GRAPH_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading graphs for topology %s", name);
  else INFO(NCCL_GRAPH, "Loading graphs");

  struct xmlHandler handlers[] = {{"graph", ncclTopoXmlGraphLoadGraph}};
  NCCLCHECK(xmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  ncclResult_t ret = ncclSuccess;
  FILE* file = fopen(xmlGraphFile, "r");
  if (file == NULL) {
    WARN("Could not open XML graph file %s : %s", xmlGraphFile, strerror(errno));
    return ncclSystemError;
  }
  struct xmlHandler handlers[] = {{"graphs", ncclTopoXmlGraphLoadGraphs}};
  xml->maxIndex = 0;
  NCCLCHECKGOTO(xmlLoadSub(file, xml, NULL, handlers, 1), ret, exit);
exit:
  fclose(file);
  return ret;
}

#elif NCCL_OS_WINDOWS

/* Stub implementations for Windows */

ncclResult_t xmlGetChar(FILE* file, char* c) {
  (void)file;
  (void)c;
  return ncclSuccess;
}

ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  (void)file;
  (void)value;
  (void)last;
  return ncclSuccess;
}

ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  (void)file;
  (void)name;
  (void)value;
  (void)last;
  return ncclSuccess;
}

ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  (void)file;
  (void)start;
  (void)next;
  return ncclSuccess;
}

ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  (void)file;
  (void)node;
  return ncclSuccess;
}

ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[],
                        int nHandlers) {
  (void)file;
  (void)xml;
  (void)head;
  (void)handlers;
  (void)nHandlers;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPciLink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadC2c(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  (void)file;
  (void)xmlGraph;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  (void)xmlGraphFile;
  (void)xml;
  return ncclSuccess;
}

#endif
