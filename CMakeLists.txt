cmake_minimum_required(VERSION 3.5)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/cmake")

include(BuildType)
include(FGInstallDirs)
include(GetPrerequisites)
include(InternalUtils)
include(platform)

option(BUILD_SHARED_LIBS        "Build shared/static library" ON)
option(FG_ENABLE_HUNTER         "Use Hunter cmake package handler" OFF)

if(FG_ENABLE_HUNTER)
    set(HUNTER_TLS_VERIFY OFF)
    set(HUNTER_BUILD_SHARED_LIBS OFF)
    include(${CMAKE_CURRENT_LIST_DIR}/CMakeModules/HunterGate.cmake)
    HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.20.60.tar.gz"
        SHA1 "35a23a3e2f8a2c87f6762c43f3cc64a0bee50943"
        FILEPATH "${CMAKE_CURRENT_LIST_DIR}/CMakeModules/Hunter/config.cmake"
    )
    mark_as_advanced(HUNTER_TLS_VERIFY)
    mark_as_advanced(HUNTER_BUILD_SHARED_LIBS)
endif()

project(Forge VERSION 1.1.0 LANGUAGES C CXX)

include(Version)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(FG_ENABLE_HUNTER)
    hunter_add_package(glbinding)
    hunter_add_package(glm)
    hunter_add_package(freetype)
    hunter_add_package(Boost)

    find_package(glbinding          REQUIRED NO_CMAKE_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    find_package(glm                REQUIRED NO_CMAKE_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    find_package(freetype CONFIG    REQUIRED NO_CMAKE_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    find_package(Boost              REQUIRED NO_CMAKE_PATH NO_SYSTEM_ENVIRONMENT_PATH)
else()
    find_package(glbinding  REQUIRED)
    find_package(glm        REQUIRED)

    # freetype.cmake builds freetype on windows
    # and calls find_package(Freetype) on other platforms
    include(freetype)

    find_package(Boost      REQUIRED)
endif()

#packages not available via hunter
find_package(Doxygen    QUIET)
find_package(FreeImage  QUIET)
if(UNIX)
    find_package(FontConfig REQUIRED)
endif()

option(FG_BUILD_DOCS
    "Build Documentation" ${DOXYGEN_FOUND})
option(FG_BUILD_EXAMPLES
    "Build Examples" ON)
option(FG_WITH_FREEIMAGE
    "Use FreeImage to add support for saving framebuffer to disk"
    ${FreeImage_FOUND})

option(FG_USE_STATIC_FREEIMAGE
    "Use static version of freeimage" OFF)
option(FG_USE_STATIC_CPPFLAGS
    "Use static libstdc++ & libgcc for generating forge library" OFF)

set(FG_USE_WINDOW_TOOLKIT "glfw3" CACHE STRING "Choose Window toolkit")
set_property(CACHE FG_USE_WINDOW_TOOLKIT PROPERTY STRINGS "glfw3" "sdl2")

mark_as_advanced(
    FG_USE_STATIC_FREEIMAGE
    FG_USE_STATIC_CPPFLAGS)

fg_deprecate(BUILD_DOCS FG_BUILD_DOCS)
fg_deprecate(BUILD_EXAMPLES FG_BUILD_EXAMPLES)
fg_deprecate(WITH_FREEIMAGE FG_WITH_FREEIMAGE)
fg_deprecate(USE_STATIC_FREEIMAGE FG_USE_STATIC_FREEIMAGE)
fg_deprecate(WITH_TOOLKIT FG_USE_WINDOW_TOOLKIT)

if(Boost_FOUND AND NOT TARGET Boost::boost)
    add_library(Boost::boost INTERFACE IMPORTED)
    set_property(TARGET Boost::boost
        PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif(Boost_FOUND AND NOT TARGET Boost::boost)

add_subdirectory(src/backend/common)
add_subdirectory(src/backend/glsl_shaders)
add_subdirectory(src/api/c)
add_subdirectory(src/api/cpp)
add_subdirectory(src/backend/opengl)

#--------------------------------------------------------------------
# Install include folder, docs, examples etc.
#--------------------------------------------------------------------
install(DIRECTORY include/
    DESTINATION ${FG_INSTALL_INC_DIR}
    COMPONENT headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE)
# The Forge version header is generated and thus need to be
# included explicitly
install(FILES ${Forge_BINARY_DIR}/include/fg/version.h
    DESTINATION "${FG_INSTALL_INC_DIR}/fg/"
    COMPONENT headers)

install(EXPORT ForgeTargets
    NAMESPACE Forge::
    DESTINATION ${FG_INSTALL_CMAKE_DIR}
    COMPONENT forge)

export(EXPORT ForgeTargets
    NAMESPACE Forge::
    FILE cmake/ForgeTargets.cmake)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${Forge_BINARY_DIR}/cmake/ForgeConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

# This config file will be installed so we need to set the install_destination
# path relative to the install path
set(INCLUDE_DIRS include)
set(CMAKE_DIR ${FG_INSTALL_CMAKE_DIR})
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/CMakeModules/ForgeConfig.cmake.in"
  "cmake/install/ForgeConfig.cmake"
  INSTALL_DESTINATION "${FG_INSTALL_CMAKE_DIR}"
  PATH_VARS INCLUDE_DIRS CMAKE_DIR)

install(FILES ${Forge_BINARY_DIR}/cmake/install/ForgeConfig.cmake
    ${Forge_BINARY_DIR}/cmake/ForgeConfigVersion.cmake
    DESTINATION ${FG_INSTALL_CMAKE_DIR}
    COMPONENT cmake)

# Following file will be used to create the config file for the build directory.
# These config files will be used by the examples to find the Forge
# libraries
set(INCLUDE_DIRS "${Forge_SOURCE_DIR}/include" "${Forge_BINARY_DIR}/include")
set(CMAKE_DIR "${Forge_BINARY_DIR}/cmake")
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/CMakeModules/ForgeConfig.cmake.in"
  "cmake/ForgeConfig.cmake"
  INSTALL_DESTINATION "${Forge_BINARY_DIR}/cmake"
  PATH_VARS INCLUDE_DIRS CMAKE_DIR
  INSTALL_PREFIX "${Forge_BINARY_DIR}")

#--------------------------------------------------------------------
# Install examples
#--------------------------------------------------------------------
# install the examples irrespective of the FG_BUILD_EXAMPLES value
# only the examples source files are installed, so the installation of these
# source files does not depend on FG_BUILD_EXAMPLES
# when FG_BUILD_EXAMPLES is OFF, the examples source is installed without
# building the example executables
install(DIRECTORY examples/ #NOTE The slash at the end is important
    DESTINATION ${FG_INSTALL_EXAMPLE_DIR}
    COMPONENT examples)

include(CPackConfig)

conditional_directory(FG_BUILD_DOCS docs)
conditional_directory(FG_BUILD_EXAMPLES examples)
