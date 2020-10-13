
set(OCCL_DPCPP_SRCS ./dpcpp_ccl.cpp ./ccl_comm_collector.cpp init.cpp)
add_library(torch_ccl_dpcpp SHARED ${OCCL_DPCPP_SRCS})
set_target_properties(torch_ccl_dpcpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(torch_ccl_dpcpp PRIVATE ${ONECCL_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${PYTORCH_INCLUDE_DIRS} ${IPEX_INCLUDE_DIRS} ../)


target_compile_options(torch_ccl_dpcpp PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -Wno-write-strings
        -Wno-unknown-pragmas
        -DOCCL_LIBNAME=libtorch_ccl_dpcpp
        )

target_compile_options(torch_ccl_dpcpp PRIVATE "-DUSE_DPCPP")
target_compile_options(torch_ccl_dpcpp PRIVATE "-fsycl")
target_compile_options(torch_ccl_dpcpp PRIVATE "-D__STRICT_ANSI__")
target_link_options(torch_ccl_dpcpp PRIVATE "-fsycl")
target_link_options(torch_ccl_dpcpp PRIVATE "-fsycl-device-code-split=per_source")

get_target_property(mpi_lib mpi IMPORTED_LOCATION)
get_filename_component(MPI_LIBRARY_DIRS ${mpi_lib} DIRECTORY)
get_target_property(fabric_lib fabric IMPORTED_LOCATION)
get_filename_component(FABRIC_LIBRARY_DIRS ${fabric_lib} DIRECTORY)
target_link_directories(torch_ccl_dpcpp PRIVATE ${MPI_LIBRARY_DIRS} ${FABRIC_LIBRARY_DIRS})

target_link_libraries(torch_ccl_dpcpp PRIVATE torch_ccl)
target_link_libraries(torch_ccl_dpcpp PRIVATE ${IPEX_LIBRARY_DIRS}/torch_ipex.so)
target_link_libraries(torch_ccl_dpcpp PRIVATE ${DEPENDS_LIB})

install(TARGETS torch_ccl_dpcpp LIBRARY DESTINATION lib OPTIONAL)