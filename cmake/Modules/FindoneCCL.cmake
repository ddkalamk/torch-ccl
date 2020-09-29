# - Try to find oneCCL
#
# The following are set after configuration is done:
#  ONECCL_FOUND          : set to true if oneCCL is found.
#  ONECCL_INCLUDE_DIR    : path to oneCCL include dir.
#  ONECCL_LIBRARIES      : list of libraries for oneCCL
#
# The following variables are used:
#  ONECCL_USE_NATIVE_ARCH : Whether native CPU instructions should be used in ONECCL. This should be turned off for
#  general packaging to avoid incompatible CPU instructions. Default: OFF.

IF (NOT ONECCL_FOUND)
SET(ONECCL_FOUND OFF)

SET(ONECCL_LIBRARIES)
SET(ONECCL_INCLUDE_DIR)

SET(ONECCL_ROOT "${PROJECT_SOURCE_DIR}/third_party/oneCCL")

ADD_SUBDIRECTORY(${ONECCL_ROOT} EXCLUDE_FROM_ALL)
IF(NOT TARGET ccl)
    MESSAGE(FATAL_ERROR "Failed to include oneDNN target")
ENDIF()
SET(ONECCL_LIBRARIES ccl)
GET_TARGET_PROPERTY(INCLUDE_DIRS ccl INCLUDE_DIRECTORIES)
SET(ONECCL_INCLUDE_DIR ${INCLUDE_DIRS})

link_directories(${PROJECT_SOURCE_DIR}/third_party/oneCCL/mpi/lib)
link_directories(${PROJECT_SOURCE_DIR}/third_party/oneCCL/ofi/lib)

MESSAGE(STATUS "ONECCL_LIBRARIES ${ONECCL_LIBRARIES}" )
MESSAGE(STATUS "ONECCL_INCLUDE_DIR ${ONECCL_INCLUDE_DIR}" )


ENDIF(NOT ONECCL_FOUND)
