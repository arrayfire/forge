CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

####################### BEGIN - Backend agnostic upstream dependencies #####################

OPTION(USE_SYSTEM_GLM "Use the system GLM library" OFF)
IF(USE_SYSTEM_GLM)
  FIND_PACKAGE(glm REQUIRED)
ELSE()
  INCLUDE(build_glm)
ENDIF()

INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIRS})

OPTION(USE_SYSTEM_FREETYPE "Use the system Freetype library" OFF)
IF(USE_SYSTEM_FREETYPE)
  FIND_PACKAGE(Freetype REQUIRED)
ELSE()
  INCLUDE(build_freetype)
ENDIF()

INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIRS})

IF(UNIX)
    FIND_PACKAGE(FontConfig REQUIRED)
    INCLUDE_DIRECTORIES("${FONTCONFIG_INCLUDE_DIR}")
ENDIF(UNIX)

FIND_PACKAGE(FreeImage REQUIRED)

INCLUDE_DIRECTORIES(${FREEIMAGE_INCLUDE_PATH})

SET(USE_WINDOW_TOOLKIT "glfw3" CACHE STRING "Choose Window toolkit")
SET_PROPERTY(CACHE USE_WINDOW_TOOLKIT PROPERTY STRINGS "glfw3" "sdl2")

IF(${USE_WINDOW_TOOLKIT} STREQUAL "glfw3")
    FIND_PACKAGE(GLFW REQUIRED)
    IF(GLFW_FOUND)
        SET(WTK_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})
        SET(WTK_LIBRARIES ${GLFW_LIBRARY})
        ADD_DEFINITIONS(-DUSE_GLFW)
    ELSE(GLFW_FOUND)
        MESSAGE(FATAL_ERROR "GLFW not found")
    ENDIF(GLFW_FOUND)
ENDIF()

IF(${USE_WINDOW_TOOLKIT} STREQUAL "sdl2")
    FIND_PACKAGE(SDL2 REQUIRED)
    IF(SDL2_FOUND)
        SET(WTK_INCLUDE_DIRS ${SDL2_INCLUDE_DIR})
        SET(WTK_LIBRARIES ${SDL2_LIBRARY})
        ADD_DEFINITIONS(-DUSE_SDL)
    ELSE(SDL2_FOUND)
        MESSAGE(FATAL_ERROR "SDL2 not found")
    ENDIF(SDL2_FOUND)
ENDIF()

####################### END - Backend agnostic upstream dependencies #######################

ADD_EXECUTABLE(glsl2cpp "${PROJECT_SOURCE_DIR}/CMakeModules/glsl2cpp.cpp")

INCLUDE(GLSLtoH)

SET(glsl_shader_headers "shader_headers")

FILE(GLOB glsl_shaders "glsl_shaders/*.glsl")

LIST(SORT glsl_shaders)

SOURCE_GROUP("src\\backend\\glsl_shaders" FILES ${glsl_shaders})

GLSL_TO_H(
    SOURCES ${glsl_shaders}
    VARNAME  shader_files
    EXTENSION "hpp"
    OUTPUT_DIR ${glsl_shader_headers}
    TARGETS glsl_shader_targets
    NAMESPACE "glsl"
    EOD "0"
    )

# DON'T REMOVE CURRENT BINARY DIRECTORY FROM
# INCLUDE_DIRECTORIES, IT IS USED TO FIND
# AUTO-GENERATED GLSL HEADERS
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

ADD_SUBDIRECTORY(opengl)