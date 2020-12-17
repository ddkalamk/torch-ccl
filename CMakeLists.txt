cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(torch_ccl)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

set(RPATH_VALUE $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${RPATH_VALUE}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

set(DEPENDS_LIB)

# Find OneCCL Lib
find_package(oneCCL REQUIRED)
list(APPEND DEPENDS_LIB ${ONECCL_LIBRARIES})

if(NOT DEFINED LIB_NAME)
    SET(LIB_NAME "torch_ccl")
endif()

add_subdirectory(./src)

function (print_configuration_summary)
    get_directory_property(CMAKE_COMPILE_DEFINITIONS DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)

    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  CMake version         : ${CMAKE_VERSION}")
    message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  Target name           : ${LIB_NAME}")
    message(STATUS "  Install path          : ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  Compile flags         : ${IPEX_COMPILE_FLAGS}")
    message(STATUS "  Compile definitions   : ${CMAKE_COMPILE_DEFINITIONS}")
    message(STATUS "  Linker options        : ${CMAKE_SHARED_LINKER_FLAGS}")
    get_target_property(LINK_LIBRARIES torch_ccl LINK_LIBRARIES)
    message(STATUS "  Linker directors        : ${LINK_LIBRARIES}")
    get_target_property(LINK_DIRECTORS torch_ccl LINK_DIRECTORIES)
    message(STATUS "  Linker libraries        : ${LINK_DIRECTORS}")

#    message(STATUS "  USE_COMPUTECPP        : ${USE_COMPUTECPP}")
#    if (USE_COMPUTECPP)
#        message(STATUS "  ComputeCpp include    : ${ComputeCpp_INCLUDE_DIRS}")
#        message(STATUS "  ComputeCpp library    : ${COMPUTECPP_RUNTIME_LIBRARY}")
#        message(STATUS "  OpenCL include        : ${OpenCL_INCLUDE_DIRS}")
#        message(STATUS "  OpenCL library        : ${OpenCL_LIBRARIES}")
#    endif(USE_COMPUTECPP)

    message(STATUS "  USE_DPCPP             : ${USE_DPCPP}")
    if (USE_DPCPP)
#        message(STATUS "  Intel SYCL include    : ${SYCL_INCLUDE_DIR}")
#        message(STATUS "  Intel SYCL library    : ${SYCL_LIBRARY}")
#        message(STATUS "  OpenCL include        : ${OpenCL_INCLUDE_DIR}")
#        message(STATUS "  OpenCL library        : ${OpenCL_LIBRARY}")
#        message(STATUS "  LevelZero include     : ${LevelZero_INCLUDE_DIRS}")
#        message(STATUS "  LevelZero library     : ${LevelZero_LIBRARIES}")
#
#        message(STATUS "  USE_ONEMKL            : ${MKLDPCPP_FOUND}")
#        if (MKLDPCPP_FOUND)
#            message(STATUS "  ONEMKL include        : ${ONEMKL_INCLUDE_DIR}")
#        endif(MKLDPCPP_FOUND)
#
#        message(STATUS "  USE_PSTL              : ${USE_PSTL}")
#        if (USE_PSTL)
#            message(STATUS "  PSTL include          : ${PSTL_INCLUDE_DIRS}")
#        endif(USE_PSTL)
#
#        message(STATUS "  USE_USM               : ${USE_USM}")
#        message(STATUS "  BUILD_BY_PER_KERNEL   : ${BUILD_BY_PER_KERNEL}")
    endif(USE_DPCPP)
    message(STATUS "")
endfunction()

print_configuration_summary()

