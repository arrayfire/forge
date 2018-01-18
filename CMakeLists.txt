cmake_minimum_required(VERSION 3.5)

OPTION(USE_HUNTER "Use Hunter cmake package handler" OFF)
if(USE_HUNTER)
    include(${CMAKE_CURRENT_LIST_DIR}/CMakeModules/HunterGate.cmake)
    
    HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.19.208.tar.gz"
        SHA1 "4128ac8c79c21b250bf825e8119ce0bff05e5132"
    )
endif()

project(Forge VERSION 1.1.0 LANGUAGES C CXX)


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}prefix;${CMAKE_PREFIX_PATH}")

include(BuildType)
include(CMakeDependentOption)
include(ConditionalDirectory)
include(DependencyCheck)
include(Version)
include(FGInstallDirs)

if(USE_HUNTER)
    hunter_add_package(glbinding)
    hunter_add_package(glm)
    hunter_add_package(freetype)
    hunter_add_package(Boost)

    find_package(freetype CONFIG REQUIRED)
else()
    find_package(freetype REQUIRED)
endif()

find_package(Doxygen QUIET)
find_package(glbinding REQUIRED)
find_package(glm REQUIRED)

if(UNIX)
    find_package(FontConfig REQUIRED)
endif()
find_package(X11 QUIET)
find_package(FreeImage QUIET)
find_package(Boost REQUIRED)

option(BUILD_DOCS      "Build Documentation"               ${DOXYGEN_FOUND})
option(BUILD_EXAMPLES  "Build Examples"                    ON)
option(WITH_FREEIMAGE  "Use FreeImage to add support for
                        saving framebuffer to disk"        ${FreeImage_FOUND})

if(Boost_FOUND AND NOT TARGET Boost::boost)
    add_library(Boost::boost INTERFACE IMPORTED)
    set_property(TARGET Boost::boost PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif(Boost_FOUND AND NOT TARGET Boost::boost)


set(WINDOW_TOOLKIT "glfw3" CACHE STRING "Choose Window toolkit")
set_property(CACHE WINDOW_TOOLKIT PROPERTY STRINGS "glfw3" "sdl2")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(src/backend/common)
add_subdirectory(src/backend/glsl_shaders)
add_subdirectory(src/api/c)
add_subdirectory(src/api/cpp)
add_subdirectory(src/backend/opengl)

conditional_directory(BUILD_DOCS docs)
conditional_directory(BUILD_EXAMPLES examples)

#--------------------------------------------------------------------
# Install include folder, docs, examples etc.
#--------------------------------------------------------------------
install(DIRECTORY include/
    DESTINATION ${FG_INSTALL_INC_DIR}
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE
    )

#--------------------------------------------------------------------
# Install examples
#--------------------------------------------------------------------
# install the examples irrespective of the BUILD_EXAMPLES value
# only the examples source files are installed, so the installation of these
# source files does not depend on BUILD_EXAMPLES
# when BUILD_EXAMPLES is OFF, the examples source is installed without
# building the example executables
install(DIRECTORY examples/ #NOTE The slash at the end is important
    DESTINATION ${FG_INSTALL_EXAMPLE_DIR}
    COMPONENT examples)
