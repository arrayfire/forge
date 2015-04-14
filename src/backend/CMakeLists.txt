CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE("${CMAKE_MODULE_PATH}/FindGLEWmx.cmake")
FIND_PACKAGE(GLFW REQUIRED)

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

IF(NOT UNIX)
  ADD_DEFINITIONS(-DAFDLL)
ENDIF()

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    "${CMAKE_SOURCE_DIR}/src/backend"
    )

IF(${GRAPHICS_FOUND})
    INCLUDE_DIRECTORIES(
        ${GLEW_INCLUDE_DIR}
        ${GLFW_INCLUDE_DIR}
        )
ENDIF(${GRAPHICS_FOUND})

FILE(GLOB backend_headers
    "*.hpp"
    "*.h")

FILE(GLOB backend_sources
    "*.cpp")

source_group(backend\\Headers FILES ${backend_headers})
source_group(backend\\Sources FILES ${backend_sources})

FILE(GLOB c_headers
    "../api/c/*.hpp"
    "../api/c/*.h"
    )

FILE(GLOB c_sources
    "../api/c/*.cpp"
    )

source_group(api\\c\\Headers FILES ${c_headers})
source_group(api\\c\\Sources FILES ${c_sources})

FILE(GLOB cpp_sources
    "../api/cpp/*.cpp"
    )

source_group(api\\cpp\\Sources FILES ${cpp_sources})

# OS Definitions
IF(UNIX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ELSE(${UNIX}) #Windows
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF()

ADD_LIBRARY(forge SHARED
           ${backend_headers}
           ${backend_sources}
           ${c_headers}
           ${c_sources}
           ${cpp_sources})

TARGET_LINK_LIBRARIES(forge
                      ${GLFW_LIBRARY}
                      ${GLEWmx_LIBRARY}
                      ${OPENGL_gl_LIBRARY}
                      ${OPENGL_glu_LIBRARY}
                      )

INSTALL(TARGETS forge DESTINATION lib)
