cmake_minimum_required(VERSION 3.5.1)

cmake_policy(VERSION 3.5)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

if(APPLE)
  # Brew does not put the glbinding cmake config files where it can be found
  set(CMAKE_PREFIX_PATH "/usr/local/opt/glbinding;${CMAKE_PREFIX_PATH}")
endif()

find_package(X11)
find_package(Doxygen)
find_package(OpenGL)
find_package(glbinding)
find_package(freetype)
find_package(glm)
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(FontConfig)
find_package(FreeImage)

find_package(GLFW)
find_package(SDL)

option(USE_HUNTER "Use Hunter cmake package handler" OFF)
option(BUILD_DOCUMENTATION "Build Documentation" ${Doxygen_FOUND})
option(BUILD_EXAMPLES "Build Examples" ON)
option(USE_FREEIMAGE "Use freeimage to allow saving of charts" ON)

option(USE_SYSTEM_GLM "Use the system GLM library" OFF)
option(USE_SYSTEM_FREETYPE "Use the system Freetype library" OFF)
option(USE_SYSTEM_GLBINDING "Use system glbinding" OFF)

set(USE_WINDOW_TOOLKIT "glfw3" CACHE STRING "Choose Window toolkit")
set_property(CACHE USE_WINDOW_TOOLKIT PROPERTY STRINGS "glfw3" "sdl2")

# TODO(umar): Add definitions should not be used. Instead use
# target_include_directories to add definitions.
if(WIN32)
  add_definitions(-DOS_WIN -DNOMINMAX)
elseif(APPLE)
  add_definitions(-DOS_MAC)
elseif(UNIX)
  add_definitions(-DOS_LNX)
endif()

add_executable(glsl2cpp "${PROJECT_SOURCE_DIR}/CMakeModules/glsl2cpp.cpp")
set_target_properties(glsl2cpp
  PROPERTIES
    CXX_STANDARD 11)

include(GLSLtoH)

set(glsl_shader_headers "shader_headers")

file(GLOB glsl_shaders "src/backend/glsl_shaders/*.glsl")

list(SORT glsl_shaders)

glsl_to_h(
    SOURCES ${glsl_shaders}
    VARNAME  shader_files
    EXTENSION "hpp"
    OUTPUT_DIR ${glsl_shader_headers}
    TARGETS glsl_shader_targets
    NAMESPACE "glsl"
    EOD "0"
    )

add_subdirectory(src/api/c)
add_subdirectory(src/api/cpp)
add_subdirectory(src/backend)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_DOCUMENTATION)
    add_subdirectory(docs)
endif(BUILD_DOCUMENTATION)
