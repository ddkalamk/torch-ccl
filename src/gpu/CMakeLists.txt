set(CCL_DPCPP_SRCS dpcpp_ccl.cpp)

set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_DEFINITIONS "USE_DPCPP;__STRICT_ANSI__")
set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-fsycl")

add_library(torch_ccl_xpu SHARED ${CCL_DPCPP_SRCS})

target_include_directories(torch_ccl_xpu PUBLIC ${ONECCL_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${PYTORCH_INCLUDE_DIRS} ${IPEX_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_directories(torch_ccl_xpu PUBLIC ${IPEX_LIBRARY_DIRS})

get_target_property(mpi_lib mpi IMPORTED_LOCATION)
get_filename_component(MPI_LIBRARY_DIRS ${mpi_lib} DIRECTORY)
get_target_property(fabric_lib fabric IMPORTED_LOCATION)
get_filename_component(FABRIC_LIBRARY_DIRS ${fabric_lib} DIRECTORY)
target_link_directories(torch_ccl_xpu PUBLIC ${MPI_LIBRARY_DIRS} ${FABRIC_LIBRARY_DIRS})
target_link_libraries(torch_ccl_xpu PUBLIC ${ONECCL_LIBRARIES})
target_link_libraries(torch_ccl_xpu PUBLIC libtorch_ipex.so)
target_link_libraries(torch_ccl_xpu PUBLIC torch_ccl)

set_target_properties(torch_ccl_xpu PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

install(TARGETS torch_ccl_xpu LIBRARY DESTINATION lib)

