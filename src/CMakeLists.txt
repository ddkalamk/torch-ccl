set(OCCL_SRCS init.cpp ProcessGroupCCL.cpp cpu/cpu_ccl.cpp dispatch_stub.cpp utils.cpp cpu/ccl_comm_collector.cpp)
add_library(torch_ccl SHARED ${OCCL_SRCS})
set_target_properties(torch_ccl PROPERTIES OUTPUT_NAME ${LIB_NAME})
set_target_properties(torch_ccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(torch_ccl PUBLIC -Wall
        -Wno-sign-compare
        -Wno-unused-function)

target_include_directories(torch_ccl PUBLIC ${ONECCL_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${PYTORCH_INCLUDE_DIRS} ${IPEX_INCLUDE_DIRS} ./)

get_target_property(mpi_lib mpi IMPORTED_LOCATION)
get_filename_component(MPI_LIBRARY_DIRS ${mpi_lib} DIRECTORY)
get_target_property(fabric_lib fabric IMPORTED_LOCATION)
get_filename_component(FABRIC_LIBRARY_DIRS ${fabric_lib} DIRECTORY)
target_link_directories(torch_ccl PUBLIC ${MPI_LIBRARY_DIRS} ${FABRIC_LIBRARY_DIRS})

target_link_directories(torch_ccl PUBLIC ${PYTORCH_LIBRARY_DIRS})
target_link_directories(torch_ccl PUBLIC ${IPEX_LIBRARY_DIRS})
target_link_libraries(torch_ccl PUBLIC ${DEPENDS_LIB})
target_link_libraries(torch_ccl PUBLIC libtorch_python.so)

install(TARGETS torch_ccl LIBRARY DESTINATION lib)
