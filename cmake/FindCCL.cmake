#
# This module defines the following variables::
#
#   CCL_FOUND          - True if CCL was found
#   CCL_INCLUDE_DIRS   - include directories for CCL
#   CCL_LIBRARIES      - link against this library to use CCL
#   CCL_VERSION_STRING - Highest supported CCL version (eg. 1.2)
#   CCL_VERSION_MAJOR  - The major version of the CCL implementation
#   CCL_VERSION_MINOR  - The minor version of the CCL implementation
#
# The module will also define two cache variables::
#
#   CCL_INCLUDE_DIR    - the CCL include directory
#   CCL_LIBRARY        - the path to the CCL library
#
#

set(CCL_INSTALL_DIR "/home/johnlu/CLionProjects/mlsl2/cmake-build-debug-clang/_installo")
#set(CCL_INSTALL_DIR "/home/johnlu/CLionProjects/mlsl2/cmake-build-debug-computecpp/_install/")
find_path(CCL_INCLUDE_DIR
  NAMES
    ccl.h ccl.hpp
  PATHS
    ${CCL_INSTALL_DIR}
  PATH_SUFFIXES
    include/cpu_gpu_dpcpp
    include)
find_library(CCL_LIBRARY
              NAMES
                ccl
              PATHS
                ${CCL_INSTALL_DIR}
              PATH_SUFFIXES
                lib/cpu_gpu_dpcpp
                lib)


set(CCL_LIBRARIES ${CCL_LIBRARY})
set(CCL_INCLUDE_DIRS ${CCL_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  CCL
  FOUND_VAR CCL_FOUND
  REQUIRED_VARS CCL_LIBRARY CCL_INCLUDE_DIR
  VERSION_VAR CL_VERSION_STRING)

mark_as_advanced(
  CCL_INCLUDE_DIR
  CCL_LIBRARY)


