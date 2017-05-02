CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

####################### BEGIN - Backend agnostic upstream dependencies #####################

if(USE_HUNTER)
    hunter_add_package(Boost COMPONENTS system filesystem)
    find_package(Boost REQUIRED COMPONENTS system filesystem)

    hunter_add_package(glm)
    find_package(glm CONFIG REQUIRED)

    hunter_add_package(freetype)
    find_package(freetype REQUIRED)
else()
    FIND_PACKAGE(Boost REQUIRED)

    OPTION(USE_SYSTEM_GLM "Use the system GLM library" OFF)
    IF(USE_SYSTEM_GLM)
      FIND_PACKAGE(glm REQUIRED)
    ELSE()
      INCLUDE(build_glm)
    ENDIF()

    OPTION(USE_SYSTEM_FREETYPE "Use the system Freetype library" OFF)
    IF(USE_SYSTEM_FREETYPE)
      FIND_PACKAGE(Freetype REQUIRED)
    ELSE()
      INCLUDE(build_freetype)
    ENDIF()
endif()

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIRS})

IF(UNIX)
    FIND_PACKAGE(FontConfig REQUIRED)
    INCLUDE_DIRECTORIES("${FONTCONFIG_INCLUDE_DIR}")
ENDIF(UNIX)

if(USE_FREEIMAGE)
	FIND_PACKAGE(FreeImage REQUIRED)

	INCLUDE_DIRECTORIES(${FREEIMAGE_INCLUDE_PATH})
	ADD_DEFINITIONS(-DUSE_FREEIMAGE)
endif()

SET(USE_WINDOW_TOOLKIT "glfw3" CACHE STRING "Choose Window toolkit")
SET_PROPERTY(CACHE USE_WINDOW_TOOLKIT PROPERTY STRINGS "glfw3" "sdl2")

IF(${USE_WINDOW_TOOLKIT} STREQUAL "glfw3")
    FIND_PACKAGE(glfw3 REQUIRED)
    IF(glfw3_FOUND)
        SET(WTK_INCLUDE_DIRS)
        SET(WTK_LIBRARIES glfw)
        ADD_DEFINITIONS(-DUSE_GLFW)
    ELSE(glfw3_FOUND)
        MESSAGE(FATAL_ERROR "GLFW not found")
    ENDIF(glfw3_FOUND)
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
