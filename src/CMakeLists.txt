function(GetOCCLLibName LIB_NAME)
    string(REPLACE "." ";" NAMES_LIST ${LIB_NAME})
    list(GET NAMES_LIST 0 OCCL_LIBNAME)
    set(OCCL_LIBNAME ${OCCL_LIBNAME} PARENT_SCOPE)
endfunction()

set(OCCL_SRCS init.cpp ProcessGroupCCL.cpp cpu/cpu_ccl.cpp dispatch_stub.cpp utils.cpp cpu/ccl_comm_collector.cpp)
add_library(torch_ccl SHARED ${OCCL_SRCS})
set_target_properties(torch_ccl PROPERTIES PREFIX "")
set_target_properties(torch_ccl PROPERTIES SUFFIX "")
set_target_properties(torch_ccl PROPERTIES OUTPUT_NAME ${LIB_NAME})
set_target_properties(torch_ccl PROPERTIES POSITION_INDEPENDENT_CODE ON)
GetOCCLLibName(${LIB_NAME})

if(DEFINED PY_MODULE)
  add_definitions(-DPY_MODULE)
endif()

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

target_include_directories(torch_ccl PRIVATE ${ONECCL_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${PYTORCH_INCLUDE_DIRS} ${IPEX_INCLUDE_DIRS} ./)

if(USE_DPCPP)
    target_compile_options(torch_ccl PRIVATE "-fsycl")
    add_subdirectory(./gpu)
endif()

get_target_property(mpi_lib mpi IMPORTED_LOCATION)
get_filename_component(MPI_LIBRARY_DIRS ${mpi_lib} DIRECTORY)
get_target_property(fabric_lib fabric IMPORTED_LOCATION)
get_filename_component(FABRIC_LIBRARY_DIRS ${fabric_lib} DIRECTORY)
target_link_directories(torch_ccl PRIVATE ${MPI_LIBRARY_DIRS} ${FABRIC_LIBRARY_DIRS})

target_link_directories(torch_ccl PRIVATE ${PYTORCH_LIBRARY_DIRS})
target_link_directories(torch_ccl PRIVATE ${IPEX_LIBRARY_DIRS})
target_link_libraries(torch_ccl PRIVATE ${DEPENDS_LIB})
target_link_libraries(torch_ccl PRIVATE libtorch_python.so)

install(TARGETS torch_ccl LIBRARY DESTINATION lib)
