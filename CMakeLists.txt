cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(torch_ccl)

# Find modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(DEPENDS_LIB)

# Find OneCCL Lib
find_package(CCL REQUIRED)
list(APPEND DEPENDS_LIB ${CCL_LIBRARY})

set(OCCL_SRCS src/init.cpp src/ProcessGroupCCL.cpp)

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

target_include_directories(torch_ccl PRIVATE ${CCL_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${PYTORCH_INCLUDE_DIRS} ${IPEX_INCLUDE_DIRS})

IF(COMPUTE_RUNTIME STREQUAL "dpcpp" OR COMPUTE_RUNTIME STREQUAL "native")
    target_include_directories(torch_ccl PRIVATE ${SYCL_INCLUDE_DIRS})
ENDIF()

if(USE_DPCPP)
    set_source_files_properties(${OCCL_SRCS} COMPILE_FLAGS "-DUSE_DPCPP ")
#    No sycl kernel so far
    set_source_files_properties(${OCCL_SRCS} COMPILE_FLAGS "-fsycl -D__STRICT_ANSI__")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl -fsycl-device-code-split=per_source")
endif()

target_link_libraries(torch_ccl PUBLIC ${DEPENDS_LIB})
link_directories(${PYTORCH_LIBRARY_DIRS})
target_link_libraries(torch_ccl PUBLIC ${PYTORCH_LIBRARY_DIRS}/libtorch_python.so)
target_link_libraries(torch_ccl PUBLIC ${IPEX_LIBRARY_DIRS}/torch_ipex.so)

#target_link_libraries(occl PUBLIC ${PYTORCH_LIBRARY_DIRS}/libc10.so)
#target_link_libraries(occl PUBLIC ${PYTORCH_LIBRARY_DIRS}/libc10d.a)
