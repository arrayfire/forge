CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

####################### BEGIN - Backend agnostic upstream dependencies #######################
# Prior to GLM 0.9.7.0, the package is found by the FindGLM.cmake module.
# This was removed with GLM 0.9.7.0, instead a glm-config.cmake configuration
# file is provided. Therefore, both FIND_PACKAGE calls are necessary.
FIND_PACKAGE(GLM QUIET)
FIND_PACKAGE(glm QUIET)

IF((NOT glm_FOUND AND NOT GLM_FOUND) OR (${USE_LOCAL_GLM}))
  SET(USE_LOCAL_GLM ON)
  MESSAGE(STATUS "Downloading GLM headers.")
  INCLUDE(build_glm)
ELSE()
  MESSAGE(STATUS "Using System GLM")
ENDIF()

INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIRS})

FIND_PACKAGE(Freetype QUIET)

IF(NOT FREETYPE_FOUND OR ${USE_LOCAL_FREETYPE})
  SET(USE_LOCAL_FREETYPE ON)
  MESSAGE(STATUS "Downloading and building Freetype libraries.")
  INCLUDE(build_freetype)
ELSE()
  MESSAGE(STATUS "Using System FreeType2")
ENDIF()

INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIRS})

IF(UNIX)
    FIND_PACKAGE(FontConfig REQUIRED)
    INCLUDE_DIRECTORIES("${FONTCONFIG_INCLUDE_DIR}")
ENDIF(UNIX)

FIND_PACKAGE(FreeImage REQUIRED)

INCLUDE_DIRECTORIES(${FREEIMAGE_INCLUDE_PATH})

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
