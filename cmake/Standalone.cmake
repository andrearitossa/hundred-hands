# Standalone/colcon entry: build the selected package and its source dependency closure.
project(${SWARM_SELECTED_PACKAGE} LANGUAGES C CXX)
set(SWARM_PYTHON OFF CACHE BOOL "Python bindings are built by the superbuild" FORCE)
get_filename_component(_swarm_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_subdirectory("${_swarm_root}" "${CMAKE_CURRENT_BINARY_DIR}/swarm")
