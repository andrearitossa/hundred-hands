cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT SWARM_IN_SUPERBUILD)
  set(SWARM_SELECTED_PACKAGE runtime)
  include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/Standalone.cmake")
  return()
endif()
# CPU-only, permissively licensed ONNX Runtime distribution.
FetchContent_Declare(ort_bin URL https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz SOURCE_SUBDIR none)
FetchContent_MakeAvailable(ort_bin)
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ort_bin_SOURCE_DIR}/lib/libonnxruntime.so
  INTERFACE_INCLUDE_DIRECTORIES ${ort_bin_SOURCE_DIR}/include)
swarm_package(runtime DEPS gnc onnxruntime)
install(FILES ${ort_bin_SOURCE_DIR}/lib/libonnxruntime.so.1.23.2 DESTINATION . RENAME libonnxruntime.so.1)
