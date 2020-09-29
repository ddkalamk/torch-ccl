cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(torch_ccl)
set(CMAKE_CXX_STANDARD 17)
# Find modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

set(DEPENDS_LIB)

# Find OneCCL Lib
find_package(oneCCL REQUIRED)
list(APPEND DEPENDS_LIB ${ONECCL_LIBRARIES})

set(OCCL_SRCS src/init.cpp src/ProcessGroupCCL.cpp src/cpu/cpu_ccl.cpp src/dispatch_stub.cpp src/utils.cpp src/ccl_comm_collector.cpp)

if(USE_DPCPP)
    set(OCCL_SRCS ${OCCL_SRCS} src/gpu/dpcpp_ccl.cpp )
endif()

IF(COMPUTE_RUNTIME STREQUAL "dpcpp" OR COMPUTE_RUNTIME STREQUAL "native")
    # import sycl SYCL
    include(cmake/SYCL.cmake)
ENDIF()

function(GetOCCLLibName LIB_NAME)
    string(REPLACE "." ";" NAMES_LIST ${LIB_NAME})
    list(GET NAMES_LIST 0 OCCL_LIBNAME)
    set(OCCL_LIBNAME ${OCCL_LIBNAME} PARENT_SCOPE)
endfunction()

add_library(torch_ccl SHARED ${OCCL_SRCS})
set_target_properties(torch_ccl PROPERTIES PREFIX "")
set_target_properties(torch_ccl PROPERTIES SUFFIX "")
set_target_properties(torch_ccl PROPERTIES OUTPUT_NAME ${LIB_NAME})
set_target_properties(torch_ccl PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(occl PROPERTIES CXX_STANDARD 17)
GetOCCLLibName(${LIB_NAME})

target_compile_options(torch_ccl PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -Wno-write-strings
        -Wno-unknown-pragmas
        -DOCCL_LIBNAME=${OCCL_LIBNAME}
        )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(torch_ccl PRIVATE -fsized-deallocation)
endif()

set(L0_SUPPORT "NO")
set(L0_ROOT "/usr/")
if(${L0_SUPPORT} STREQUAL "YES")

    #    set(CMAKE_CLANG_FLAGS "${CMAKE_CLANG_FLAGS} -Wno-deprecated-declarations")
    #    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

    #    LIST(APPEND CMAKE_MODULE_PATH "${L0_ROOT}/lib/cmake/level_zero" "${L0_ROOT}/lib/cmake/level_zero")
    #find_package(LevelZero CONFIG REQUIRED PATHS ${CMAKE_MODULE_PATH})
    list(APPEND CCL_INCLUDE_DIR "${L0_ROOT}/include/level_zero")
#    list(APPEND CCL_INCLUDE_DIR "/opt/intel/opencl/include/")
#    list(APPEND CCL_INCLUDE_DIR "/p/pdsd/scratch/sivanov/opencl")

    #"/p/pdsd/scratch/sivanov/dpcpp/build/linux_prod/compiler/linux/lib/clang/10.0.0/include")

    #    link_directories("/usr/local/lib/level_zero/")
    #list(APPEND CMAKE_SHARED_LINKER_FLAGS "-L${L0_ROOT}/lib/level_zero/")# -lxe_loader")
    #    list(APPEND EXTERNAL_LIBS "-llevel_zero")
    #list(APPEND CMAKE_STATIC_LINKER_FLAGS "-llevel_zero")
    #    add_subdirectory(examples/level_zero)


    #    option(CCL_GPU_DEVICES_AFFINITY_ENABLE "Enable L0" ON)
    #    if(CCL_GPU_DEVICES_AFFINITY_ENABLE)
    #        set(CCL_GPU_DEVICES_AFFINITY_MASK_SIZE 4)
    #        message ("Set L0 device mask affinity size: ${CCL_GPU_DEVICES_AFFINITY_MASK_SIZE}")
    #    endif()

endif()

target_include_directories(torch_ccl PRIVATE ${ONECCL_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${PYTORCH_INCLUDE_DIRS} ${IPEX_INCLUDE_DIRS} ./src/)

if(USE_DPCPP)
    set_source_files_properties(${OCCL_SRCS} COMPILE_FLAGS "-DUSE_DPCPP ")
    set_source_files_properties(${OCCL_SRCS} COMPILE_FLAGS "-fsycl -D__STRICT_ANSI__")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl -fsycl-device-code-split=per_source")
endif()

target_link_libraries(torch_ccl PRIVATE ${DEPENDS_LIB})
link_directories(${PYTORCH_LIBRARY_DIRS})
target_link_libraries(torch_ccl PUBLIC ${PYTORCH_LIBRARY_DIRS}/libtorch_python.so)
target_link_libraries(torch_ccl PUBLIC ${IPEX_LIBRARY_DIRS}/torch_ipex.so)


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
    message(STATUS "  Linker libraries        : ${LINK_LIBRARIES}")
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

