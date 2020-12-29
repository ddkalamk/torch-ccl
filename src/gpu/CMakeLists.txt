
set(CCL_DPCPP_SRCS ./dpcpp_ccl.cpp ./ccl_comm_collector.cpp)
#target_include_directories(torch_ccl_dpcpp PRIVATE ${ONECCL_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${PYTORCH_INCLUDE_DIRS} ${IPEX_INCLUDE_DIRS} ../)


#target_compile_options(torch_ccl_dpcpp PRIVATE
#        -Wall
#        -Wextra
#        -Wno-unused-parameter
#        -Wno-missing-field-initializers
#        -Wno-write-strings
#        -Wno-unknown-pragmas
#        -DOCCL_LIBNAME=libtorch_ccl_dpcpp
#        )

set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-DUSE_DPCPP")
set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-fsycl")
set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-D__STRICT_ANSI__")

target_sources(torch_ccl PUBLIC ${CCL_DPCPP_SRCS})
target_link_libraries(torch_ccl PUBLIC libtorch_ipex.so)
#target_link_options(torch_ccl_dpcpp PRIVATE "-fsycl")
#target_link_options(torch_ccl_dpcpp PRIVATE "-fsycl-device-code-split=per_source")

