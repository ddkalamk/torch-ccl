cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(occl)

# Find modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(DEPENDS_LIB)

# Find OneCCL Lib
find_package(CCL REQUIRED)
list(APPEND DEPENDS_LIB ${CCL_LIBRARY})

set(OCCL_SRCS src/init.cpp src/ProcessGroupCCL.cpp)

IF(USE_SYCL)
    # Find SYCL
    include(cmake/DPCPP.cmake)
ENDIF()

add_library(occl SHARED ${OCCL_SRCS})
set_target_properties(occl PROPERTIES PREFIX "")
set_target_properties(occl PROPERTIES SUFFIX "")
set_target_properties(occl PROPERTIES OUTPUT_NAME ${LIB_NAME})
set_target_properties(occl PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(occl PROPERTIES CXX_STANDARD 17)

target_compile_options(occl PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -Wno-write-strings
        -Wno-unknown-pragmas
        )

set(L0_SUPPORT "YES")
set(L0_ROOT "/usr/")
if(${L0_SUPPORT} STREQUAL "YES")

    #    set(CMAKE_CLANG_FLAGS "${CMAKE_CLANG_FLAGS} -Wno-deprecated-declarations")
    #    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

    #    LIST(APPEND CMAKE_MODULE_PATH "${L0_ROOT}/lib/cmake/level_zero" "${L0_ROOT}/lib/cmake/level_zero")
    #find_package(LevelZero CONFIG REQUIRED PATHS ${CMAKE_MODULE_PATH})
    message("Level0 examples, path: ${CMAKE_MODULE_PATH}")
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

target_include_directories(occl PRIVATE ${CCL_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${PYTORCH_INCLUDE_DIRS} ${IPEX_INCLUDE_DIRS})

IF(USE_SYCL)
    MESSAGE(STATUS here johnlu)
    target_include_directories(occl PRIVATE ${SYCL_INCLUDE_DIRS})
ENDIF()

if(USE_DPCPP)
    set_source_files_properties(${OCCL_SRCS} COMPILE_FLAGS "-DUSE_DPCPP ")
#    No sycl kernel so far
    set_source_files_properties(${OCCL_SRCS} COMPILE_FLAGS "-fsycl -D__STRICT_ANSI__")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl -fsycl-device-code-split=per_source")
endif()

target_link_libraries(occl PUBLIC ${DEPENDS_LIB})
link_directories(${PYTORCH_LIBRARY_DIRS})
target_link_libraries(occl PUBLIC ${PYTORCH_LIBRARY_DIRS}/libtorch_python.so)
target_link_libraries(occl PUBLIC ${IPEX_LIBRARY_DIRS}/torch_ipex.so)

#target_link_libraries(occl PUBLIC ${PYTORCH_LIBRARY_DIRS}/libc10.so)
#target_link_libraries(occl PUBLIC ${PYTORCH_LIBRARY_DIRS}/libc10d.a)
