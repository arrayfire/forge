CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Forge-Examples)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

OPTION(BUILD_EXAMPLES_CUDA "Turn off/on building cuda examples" ON)
OPTION(BUILD_EXAMPLES_OPENCL "Turn off/on building opencl examples" ON)
OPTION(USE_SYSTEM_CL2HPP "Use cl2.hpp header installed on the system" OFF)

IF(NOT USE_SYSTEM_CL2HPP)
    INCLUDE(build_cl2hpp)
ENDIF()

MACRO(BUILD_EXAMPLE EX_NAME EX_SRC COMPUTE_NAME FG_LIBS COMPUTE_LIBS)
    IF(${COMPUTE_NAME} STREQUAL "cuda")
        CUDA_ADD_EXECUTABLE(example_${EX_NAME}_${COMPUTE_NAME} ${EX_SRC})
    ELSE()
        ADD_EXECUTABLE(example_${EX_NAME}_${COMPUTE_NAME} ${EX_SRC})
    ENDIF()
    TARGET_LINK_LIBRARIES(example_${EX_NAME}_${COMPUTE_NAME} ${COMPUTE_LIBS} ${FG_LIBS})
    SET_TARGET_PROPERTIES(example_${EX_NAME}_${COMPUTE_NAME}
        PROPERTIES
        OUTPUT_NAME ${EX_NAME}_${COMPUTE_NAME}
        RUNTIME_OUTPUT_DIRECTORY ${DIR_NAME}
        FOLDER "Examples/${COMPUTE_NAME}")
    IF(${COMPUTE_NAME} STREQUAL "opencl")
        IF(NOT USE_SYSTEM_CL2HPP)
            ADD_DEPENDENCIES(example_${EX_NAME}_${COMPUTE_NAME} cl2hpp)
        ENDIF()
    ENDIF()
ENDMACRO()

IF(TARGET forge)

    # Building with Forge Source
    SET(FORGE_FOUND False)

ELSE(TARGET forge)

    SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
    FIND_PACKAGE(Forge REQUIRED)

    SET(X11_LIBS "")

    INCLUDE_DIRECTORIES(${FORGE_INCLUDE_DIRS})

    # OS Definitions
    IF(UNIX)
        ADD_DEFINITIONS(-Wall -std=c++11 -fvisibility=hidden)

        # GCC 5.3 and above give errors for mempcy from <string.h>
        # This is a (temporary) fix for that
        IF("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.3.0")
            ADD_DEFINITIONS(-D_FORCE_INLINES)
        ENDIF()

        # MacOS specific environment settings
        IF(APPLE)
            ADD_DEFINITIONS(-DOS_MAC)
            SET(CMAKE_MACOSX_RPATH ON)
            SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
            SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
            SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
            SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

            LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}" isSystemDir)
            IF("${isSystemDir}" STREQUAL "-1")
                SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
            ENDIF("${isSystemDir}" STREQUAL "-1")

            FIND_PACKAGE(X11 REQUIRED)
            INCLUDE_DIRECTORIES(${X11_INCLUDE_DIR})

        ELSE(APPLE)
            ADD_DEFINITIONS(-DOS_LNX)
        ENDIF(APPLE)
    ELSE(UNIX)
        ADD_DEFINITIONS(-DOS_WIN -DNOMINMAX)
    ENDIF(UNIX)
ENDIF(TARGET forge)

IF(UNIX)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-function")
ENDIF()

FILE(GLOB CPU_EXAMPLE_SRC_FILES "cpu/*.cpp")
LIST(SORT CPU_EXAMPLE_SRC_FILES)
FOREACH(FILE ${CPU_EXAMPLE_SRC_FILES})
    GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
    GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
    GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

    IF(TARGET forge)
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} forge "")
    ELSE(TARGET forge)
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} ${FORGE_LIBRARIES} "${X11_LIBS}")
    ENDIF()
ENDFOREACH()

IF (BUILD_EXAMPLES_CUDA)
    FIND_PACKAGE(CUDA QUIET)
    IF(CUDA_FOUND)

        IF(UNIX)
            # GCC 5.3 and above give errors for mempcy from <string.h>
            # This is a (temporary) fix for that
            IF("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.3.0")
                ADD_DEFINITIONS(-D_FORCE_INLINES)
            ENDIF()
            # GCC 6.0 and above default to g++14, enabling c++11 features by default
            # Enabling c++11 with nvcc 7.5 + gcc 6.x doesn't seem to work
            # Only solution for now is to force use c++03 for gcc 6.x
            IF("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "6.0.0")
                SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -std=c++98")
            ENDIF()
        ENDIF(UNIX)

        INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

        FILE(GLOB CUDA_EXAMPLE_SRC_FILES "cuda/*.cu")
        LIST(SORT CUDA_EXAMPLE_SRC_FILES)
        FOREACH(FILE ${CUDA_EXAMPLE_SRC_FILES})
            GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
            GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
            GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

            IF(TARGET forge)
                BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} forge "${CUDA_LIBRARIES}")
            ELSE(TARGET forge)
                BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} ${FORGE_LIBRARIES}
                    "${X11_LIBS};${CUDA_LIBRARIES}")
            ENDIF()
        ENDFOREACH()
    ELSE()
        MESSAGE(STATUS "CUDA Toolkit not found, not building CUDA examples.")
    ENDIF()
ENDIF()

IF (BUILD_EXAMPLES_OPENCL)
    FIND_PACKAGE(OpenCL QUIET)
    IF(OpenCL_FOUND)
        FILE(GLOB OpenCL_EXAMPLE_SRC_FILES "opencl/*.cpp")
        LIST(SORT OpenCL_EXAMPLE_SRC_FILES)
        INCLUDE_DIRECTORIES(
            "${CMAKE_CURRENT_SOURCE_DIR}/opencl"
            ${OpenCL_INCLUDE_DIRS}
            ${CL2HPP_INCLUDE_DIRECTORY}
            )
        FOREACH(FILE ${OpenCL_EXAMPLE_SRC_FILES})
            GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
            GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
            GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

            IF(TARGET forge)
                BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} forge "${OpenCL_LIBRARIES}")
            ELSE(TARGET forge)
                BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} ${FORGE_LIBRARIES}
                    "${X11_LIBS};${OpenCL_LIBRARIES}")
            ENDIF()
        ENDFOREACH()
    ELSE()
        MESSAGE(STATUS "OpenCL Libraries not found, not building OpenCL examples.")
    ENDIF()
ENDIF()

MARK_AS_ADVANCED(BUILD_EXAMPLES_CUDA BUILD_EXAMPLES_OPENCL)
