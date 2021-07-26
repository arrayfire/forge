cmake_minimum_required(VERSION 3.5)

project(Forge VERSION 1.0.8 LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMakeModules")
list(APPEND CMAKE_PREFIX_PATH "${Forge_BINARY_DIR}/cmake")

include(ForgeBuildType)
include(ForgeInstallDirs)
include(ForgeInternalUtils)
include(ForgePlatformSetup)
include(ForgeVersion)
include(ForgeConfigureDepsVars)

set_policies(TYPE NEW POLICIES CMP0072 CMP0077)

option(BUILD_SHARED_LIBS "Build shared/static library" ON)

find_package(Boost      REQUIRED)
find_package(Doxygen    QUIET)
find_package(FontConfig QUIET)
find_package(FreeImage  QUIET)
find_package(Freetype   REQUIRED)
find_package(Sphinx     QUIET)
find_package(glad CONFIG QUIET)
find_package(glm CONFIG QUIET)

if(UNIX)
    dependency_check(FontConfig_FOUND
        "FontConfig is required on non-windows OS")
endif()

option(FG_BUILD_DOCS
    "Build Documentation" $<AND:${DOXYGEN_FOUND},${Sphinx_FOUND}>)
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

set(FG_RENDERING_BACKEND "OpenGL" CACHE STRING "Choose Rendering Backend")
set_property(CACHE FG_RENDERING_BACKEND PROPERTY STRINGS "OpenGL")

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

if(NOT TARGET glad::glad) # find_package(glad) failed
    fg_dep_check_and_populate(${glad_prefix}
        URI https://github.com/arrayfire/glad.git
        REF obj_lib
    )
    add_subdirectory(${${glad_prefix}_SOURCE_DIR} ${${glad_prefix}_BINARY_DIR})

    add_library(forge_glad STATIC $<TARGET_OBJECTS:glad_obj_lib>)
    target_link_libraries(forge_glad PUBLIC ${CMAKE_DL_LIBS})
    target_include_directories(forge_glad
        PUBLIC
        $<BUILD_INTERFACE:$<TARGET_PROPERTY:glad_obj_lib,INTERFACE_INCLUDE_DIRECTORIES>>
        )
else()
    add_library(forge_glad ALIAS glad::glad)
endif()

add_library(forge_glm INTERFACE)
if(TARGET glm::glm)
    target_include_directories(forge_glm
        SYSTEM INTERFACE
        $<TARGET_PROPERTY:glm::glm,INTERFACE_INCLUDE_DIRECTORIES>
    )
else() # find_package(glm) failed
    fg_dep_check_and_populate(${glm_prefix}
        URI https://github.com/g-truc/glm.git
        REF 0.9.9.8
    )
    target_include_directories(forge_glm INTERFACE "${${glm_prefix}_SOURCE_DIR}")
endif()

add_subdirectory(src/backend/common)
add_subdirectory(src/backend/glsl_shaders)
add_subdirectory(src/api/c)
add_subdirectory(src/api/cpp)
add_subdirectory(src/backend)

#--------------------------------------------------------------------
# Install include folder, docs, examples etc.
#--------------------------------------------------------------------
install(DIRECTORY include/
    DESTINATION ${FG_INSTALL_INC_DIR}
    COMPONENT forge_dev
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE)

install(FILES ${Forge_BINARY_DIR}/include/fg/version.h
    DESTINATION "${FG_INSTALL_INC_DIR}/fg/"
    COMPONENT forge_dev)

# install the examples irrespective of the FG_BUILD_EXAMPLES value
# only the examples source files are installed, so the installation of these
# source files does not depend on FG_BUILD_EXAMPLES
# when FG_BUILD_EXAMPLES is OFF, the examples source is installed without
# building the example executables
install(DIRECTORY examples/ #NOTE The slash at the end is important
    DESTINATION ${FG_INSTALL_EXAMPLE_DIR}
    COMPONENT forge_dev)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${Forge_BINARY_DIR}/ForgeConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)


# export install config file
set(INCLUDE_DIRS include)
set(CMAKE_DIR ${FG_INSTALL_CMAKE_DIR})
configure_package_config_file(
    "${Forge_SOURCE_DIR}/CMakeModules/ForgeConfig.cmake.in"
    "cmake_install/ForgeConfig.cmake"
    INSTALL_DESTINATION "${FG_INSTALL_CMAKE_DIR}"
    PATH_VARS INCLUDE_DIRS CMAKE_DIR
)
install(FILES ${Forge_BINARY_DIR}/cmake_install/ForgeConfig.cmake
    ${Forge_BINARY_DIR}/ForgeConfigVersion.cmake
    DESTINATION ${FG_INSTALL_CMAKE_DIR}
    COMPONENT forge_dev
)
install(EXPORT ForgeTargets
    NAMESPACE Forge::
    DESTINATION ${FG_INSTALL_CMAKE_DIR}
    COMPONENT forge_dev
)

# export build tree targets config file
set(INCLUDE_DIRS "${Forge_SOURCE_DIR}/include" "${Forge_BINARY_DIR}/include")
set(CMAKE_DIR "${Forge_BINARY_DIR}")
configure_package_config_file(
    "${Forge_SOURCE_DIR}/CMakeModules/ForgeConfig.cmake.in"
    "ForgeConfig.cmake"
    INSTALL_DESTINATION "${Forge_BINARY_DIR}"
    PATH_VARS INCLUDE_DIRS CMAKE_DIR
    INSTALL_PREFIX "${Forge_BINARY_DIR}"
)
export(EXPORT ForgeTargets
    NAMESPACE Forge::
    FILE ForgeTargets.cmake
)

conditional_directory(FG_BUILD_DOCS docs)
conditional_directory(FG_BUILD_EXAMPLES examples)

mark_as_advanced(
    pkgcfg_lib_FontConfigPkg_freetype
    pkgcfg_lib_FontConfigPkg_fontconfig
    Boost_INCLUDE_DIR
    SPHINX_EXECUTABLE
    VCPKG_APPLOCAL_DEPS
    VCPKG_BOOTSTRAP_OPTIONS
    VCPKG_INSTALL_OPTIONS
    VCPKG_MANIFEST_DIR
    VCPKG_MANIFEST_INSTALL
    VCPKG_MANIFEST_MODE
    VCPKG_OVERLAY_PORTS
    VCPKG_OVERLAY_TRIPLETS
    VCPKG_TARGET_TRIPLET
    X_VCPKG_APPLOCAL_DEPS_INSTALL
    X_VCPKG_APPLOCAL_DEPS_SERIALIZED
    Z_VCPKG_BUILTIN_POWERSHELL_PATH
    Z_VCPKG_PWSH_PATH
    Z_VCPKG_CL
    _VCPKG_INSTALLED_DIR
    glm_DIR
    glad_DIR
)

include(ForgeCPackConfig)
