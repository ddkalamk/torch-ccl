
set(CCL_DPCPP_SRCS dpcpp_ccl.cpp)

set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-DUSE_DPCPP")
set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-fsycl")
set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-D__STRICT_ANSI__")

target_sources(torch_ccl PUBLIC ${CCL_DPCPP_SRCS})
target_link_libraries(torch_ccl PUBLIC libtorch_ipex.so)

