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

OPTION(USE_SYSTEM_GLM "Use system GLM" OFF)
IF(USE_SYSTEM_GLM)
      FIND_PACKAGE(GLM REQUIRED)
  ELSE()
      INCLUDE("${CMAKE_MODULE_PATH}/build_glm.cmake")
ENDIF()

INCLUDE("${CMAKE_MODULE_PATH}/build_freetypegl.cmake")

IF(NOT UNIX)
  ADD_DEFINITIONS(-DAFDLL)
ENDIF()

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    "${CMAKE_SOURCE_DIR}/src"
    ${GLM_INCLUDE_DIRS}
    "${FREETYPEGL_INCLUDE_DIRS}"
    )

IF(${GRAPHICS_FOUND})
    INCLUDE_DIRECTORIES(
        ${GLEW_INCLUDE_DIR}
        ${GLFW_INCLUDE_DIR}
        )
ENDIF(${GRAPHICS_FOUND})


FILE(GLOB api_headers
    "../include/*.h"
    )

source_group(API FILES ${api_headers})

FILE(GLOB headers
    "*.hpp"
    "*.h"
    )

FILE(GLOB sources
    "*.c"
    "*.cpp"
    )

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

# OS Definitions
IF(UNIX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ELSE(${UNIX}) #Windows
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF()

ADD_LIBRARY(forge SHARED
           ${headers}
           ${sources})

TARGET_LINK_LIBRARIES(forge
                      ${GLFW_LIBRARY}
                      ${GLEWmx_LIBRARY}
                      ${OPENGL_gl_LIBRARY}
                      ${OPENGL_glu_LIBRARY}
                      ${FREETYPEGL_LIBRARIES}
                      )

INSTALL(TARGETS forge DESTINATION lib)
