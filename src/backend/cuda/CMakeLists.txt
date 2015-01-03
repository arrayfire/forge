CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE("${CMAKE_MODULE_PATH}/FindGLEWmx.cmake")
FIND_PACKAGE(GLFW)

IF(GLFW_FOUND AND GLEWmx_FOUND AND OPENGL_FOUND)
  SET(GRAPHICS_FOUND ON)
  ADD_DEFINITIONS(-DGLEW_MX -DWITH_GRAPHICS)
ELSE(GLFW_FOUND AND GLEWmx_FOUND AND OPENGL_FOUND)
    IF(NOT GLFW_FOUND)
        MESSAGE(FATAL_ERROR "GLFW Not Found")
    ELSEIF(NOT GLEWmx_FOUND)
        MESSAGE(FATAL_ERROR "GLEW-MX Not Found")
    ELSEIF(NOT OPENGL_FOUND)
        MESSAGE(FATAL_ERROR "OpenGL Not Found")
    ENDIF()
ENDIF(GLFW_FOUND AND GLEWmx_FOUND AND OPENGL_FOUND)

# Disables running cuda_compute_check.c when build windows using remote
IF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  INCLUDE("${CMAKE_MODULE_PATH}/CUDACheckCompute.cmake")
ELSE(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  IF(NOT DEFINED CUDA_GENERATE_CODE)
    SET(CUDA_GENERATE_CODE "arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}")
  ENDIF(NOT DEFINED CUDA_GENERATE_CODE)

  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch compute_${CUDA_COMPUTE_CAPABILITY})
ENDIF()

IF(NOT UNIX)
  ADD_DEFINITIONS(-DAFDLL)
ENDIF()

ADD_DEFINITIONS(-DAF_CUDA)
REMOVE_DEFINITIONS(-std=c++0x)

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/src/backend/cuda"
    )

INCLUDE_DIRECTORIES(
    ${GLEW_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIR}
    )

FILE(GLOB cuda_headers
     "*.hpp"
     "*.h")

FILE(GLOB cuda_sources
    "*.cu"
    "*.cpp")

source_group(backend\\cuda\\Headers FILES ${cuda_headers})
source_group(backend\\cuda\\Sources FILES ${cuda_sources})

FILE(GLOB backend_headers
    "../*.hpp"
    "../*.h"
    )

FILE(GLOB backend_sources
    "../*.cpp"
    )

source_group(backend\\Headers FILES ${backend_headers})
source_group(backend\\Sources FILES ${backend_sources})

FILE(GLOB c_headers
    "../../api/c/*.hpp"
    "../../api/c/*.h"
    )

FILE(GLOB c_sources
    "../../api/c/*.cpp"
    )

source_group(api\\c\\Headers FILES ${c_headers})
source_group(api\\c\\Sources FILES ${c_sources})

FILE(GLOB cpp_sources
    "../../api/cpp/*.cpp"
    )

source_group(api\\cpp\\Sources FILES ${cpp_sources})

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND "${APPLE}")
    ADD_DEFINITIONS(-stdlib=libstdc++)
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
    SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
    SET(CUDA_HOST_COMPILER "/usr/bin/clang++")
ENDIF()

CUDA_ADD_LIBRARY(fwcuda SHARED
                ${cuda_headers}
                ${cuda_sources}
                ${backend_headers}
                ${backend_sources}
                ${c_headers}
                ${c_sources}
                ${cpp_sources}
                )

TARGET_LINK_LIBRARIES(fwcuda
                     ${CUDA_LIBRARIES}
                     ${CUDA_CUDA_LIBRARY})

TARGET_LINK_LIBRARIES(fwcuda
                     ${GLFW_LIBRARY}
                     ${GLEWmx_LIBRARY}
                     ${OPENGL_gl_LIBRARY}
                     ${OPENGL_glu_LIBRARY})

INSTALL(TARGETS fwcuda DESTINATION lib)
