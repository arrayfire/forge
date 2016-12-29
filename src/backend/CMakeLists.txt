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
