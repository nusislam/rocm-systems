# Graph sources
set(GRAPH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/topo.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/xml.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/search.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/paths.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/connect.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rings.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/trees.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/discovery/topo_discovery.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/discovery/topo_discovery_dev.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/discovery/topo_discovery_net.cc
)

# Add graph sources to parent scope
set(GRAPH_SOURCES ${GRAPH_SOURCES} PARENT_SCOPE)
