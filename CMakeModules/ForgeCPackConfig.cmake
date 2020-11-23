# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/nsis")

include(ForgeVersion)
include(CPackIFW)

set(VENDOR_NAME "ArrayFire")
set(APP_NAME "Forge")
string(TOLOWER ${APP_NAME} LC_APP_NAME)
set(APP_URL "https://github.com/arrayfire/forge")

# Long description of the package
set(CPACK_PACKAGE_DESCRIPTION
"Forge is an OpenGL interop library that can be used with ArrayFire or any other application using CUDA or
OpenCL compute backend. The goal of **Forge** is to provide high performance OpenGL visualizations
for C/C++ applications that use CUDA/OpenCL. Forge uses OpenGL >=3.3 forward compatible contexts, so
please make sure you have capable hardware before trying it out.")

# Short description of the package
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
  "A high performance visualization library.")

# Common settings to all packaging tools
set(CPACK_PREFIX_DIR ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_NAME "${APP_NAME}")
set(CPACK_PACKAGE_VENDOR "${VENDOR_NAME}")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY ${APP_NAME})
set(CPACK_PACKAGE_CONTACT "ArrayFire Development Group <technical@arrayfire.com>")
set(MY_CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/assets/arrayfire.ico")

file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/assets/" NATIVE_ASSETS_PATH)
string(REPLACE "\\" "\\\\" NATIVE_ASSETS_PATH  ${NATIVE_ASSETS_PATH})
set(CPACK_FG_ASSETS_DIR "${NATIVE_ASSETS_PATH}")

set(CPACK_PACKAGE_VERSION ${Forge_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR "${Forge_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Forge_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${Forge_VERSION_PATCH}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${APP_NAME}")
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION})

# Platform specific settings for CPACK generators
# - OSX specific
#   - productbuild (OSX only)
# - Windows
#   - NSIS64 Generator
if(WIN32)
  set(WIN_INSTALL_SOURCE ${PROJECT_SOURCE_DIR}/CMakeModules/nsis)

  set(LICENSE_FILE       "${Forge_SOURCE_DIR}/.github/LICENSE")
  set(LICENSE_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/license.txt")
  configure_file(${LICENSE_FILE} ${LICENSE_FILE_OUT})
  set(CPACK_RESOURCE_FILE_LICENSE ${LICENSE_FILE_OUT})

  #NSIS SPECIFIC VARIABLES
  set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
  set(CPACK_NSIS_MODIFY_PATH ON)
  set(CPACK_NSIS_DISPLAY_NAME "${APP_NAME}")
  set(CPACK_NSIS_PACKAGE_NAME "${APP_NAME}")
  set(CPACK_NSIS_HELP_LINK "${APP_URL}")
  set(CPACK_NSIS_URL_INFO_ABOUT "${APP_URL}")
  set(CPACK_NSIS_INSTALLED_ICON_NAME "${MY_CPACK_PACKAGE_ICON}")
  if(CMAKE_CL_64)
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
  else(CMAKE_CL_64)
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
  endif(CMAKE_CL_64)
else()
    set(CPACK_RESOURCE_FILE_LICENSE "${Forge_SOURCE_DIR}/.github/LICENSE")
    set(CPACK_RESOURCE_FILE_README "${Forge_SOURCE_DIR}/README.md")
endif()

# Set the default components installed in the package
#get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)

include(CPackComponent)

cpack_add_install_type(Development
  DISPLAY_NAME "Development")
cpack_add_install_type(Runtime
  DISPLAY_NAME "Runtime")

cpack_add_component_group(backends
  DISPLAY_NAME "Forge"
  DESCRIPTION "Forge libraries")

# This component usually used for generating graphical installers where upstream
# dependencies aren't usually managed by some sort of global package manager.
# This is typically the case with Windows installers
if(WIN32)
  cpack_add_component(forge_dependencies
    DISPLAY_NAME "Forge Dependencies"
    DESCRIPTION "Libraries required by Forge OpenGL backend"
    PARENT_GROUP backends
    INSTALL_TYPES Development Runtime)
endif()

cpack_add_component(forge
  DISPLAY_NAME "Forge"
  DESCRIPTION "Forge library."
  PARENT_GROUP backends
  INSTALL_TYPES Development Runtime)

cpack_add_component(forge_dev
  DISPLAY_NAME "Development files required for forge"
  DESCRIPTION "Development files include headers,
               cmake config files and example source files
               apart from runtime libraries."
  INSTALL_TYPES Development)

##
# IFW CPACK generator
# Uses Qt installer framework, cross platform installer generator.
# Uniform installer GUI on all major desktop platforms: Windows, OSX & Linux.
##
set(CPACK_IFW_PACKAGE_TITLE "${CPACK_PACKAGE_NAME}")
set(CPACK_IFW_PACKAGE_PUBLISHER "${CPACK_PACKAGE_VENDOR}")
set(CPACK_IFW_PRODUCT_URL "${APP_URL}")
set(CPACK_IFW_PACKAGE_ICON "${MY_CPACK_PACKAGE_ICON}")
set(CPACK_IFW_PACKAGE_WINDOW_ICON "${CMAKE_SOURCE_DIR}/assets/arrayfire_icon.png")
set(CPACK_IFW_PACKAGE_LOGO "${CMAKE_SOURCE_DIR}/assets/arrayfire_logo.png")
if (WIN32)
  set(CPACK_IFW_ADMIN_TARGET_DIRECTORY "$PROGRAMFILES64/${CPACK_PACKAGE_INSTALL_DIRECTORY}")
else ()
  set(CPACK_IFW_ADMIN_TARGET_DIRECTORY "/opt/${CPACK_PACKAGE_INSTALL_DIRECTORY}")
endif ()
cpack_ifw_configure_component_group(backends)
if(WIN32)
  cpack_ifw_configure_component(forge_dependencies)
endif()
cpack_ifw_configure_component(forge)
cpack_ifw_configure_component(forge_dev)

##
# Debian package
##
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CMAKE_INSTALL_RPATH "/usr/lib;${Forge_BINARY_DIR}/src/backend")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE http://www.arrayfire.com)
set(CPACK_DEBIAN_FORGE_PACKAGE_NAME "lib${LC_APP_NAME}")
set(CPACK_DEBIAN_FORGE_DEV_PACKAGE_NAME "lib${LC_APP_NAME}-dev")
## libfreetype6 isn't explicitly mentioned as it is dependency of libfontconfig1
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libglfw3 (>= 3.0), libfreeimage3 (>= 3.15), libfontconfig1 (>= 2.11)")
set(CPACK_DEBIAN_FORGE_DEV_PACKAGE_DEPENDS "${CPACK_DEBIAN_FORGE_PACKAGE_NAME} (= ${CPACK_PACKAGE_VERSION})")

##
# RPM package
##
set(CPACK_RPM_PACKAGE_AUTOREQ NO)
set(CPACK_RPM_FORGE_PACKAGE_NAME "${LC_APP_NAME}")
set(CPACK_RPM_FORGE_DEV_PACKAGE_NAME "${LC_APP_NAME}-devel")
set(CPACK_RPM_FILE_NAME "%{name}-%{version}-%{release}%{?dist}_%{_arch}.rpm")
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_PACKAGE_LICENSE "BSD")
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_PACKAGE_URL "${APP_URL}")
## freetype isn't explicitly mentioned as it is dependency of fontconfig
set(CPACK_RPM_PACKAGE_REQUIRES "glfw >= 3.2, freeimage >= 3.15, fontconfig >= 2.11")
set(CPACK_RPM_FORGE_DEV_PACKAGE_REQUIRES "${CPACK_RPM_FORGE_PACKAGE_NAME} == ${CPACK_PACKAGE_VERSION}")

##
# Source package
##
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}_src_${CPACK_PACKAGE_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR})
set(CPACK_SOURCE_IGNORE_FILES
    "/build"
    "CMakeFiles"
    "/\\\\.dir"
    "/\\\\.git"
    "/\\\\.gitignore$"
    ".*~$"
    "\\\\.bak$"
    "\\\\.swp$"
    "\\\\.orig$"
    "/\\\\.DS_Store$"
    "/Thumbs\\\\.db"
    "/CMakeLists.txt.user$"
    ${CPACK_SOURCE_IGNORE_FILES})
# Ignore build directories that may be in the source tree
file(GLOB_RECURSE CACHES "${CMAKE_SOURCE_DIR}/CMakeCache.txt")

include(CPack)

if (WIN32)
    # Configure file with custom definitions for NSIS.
    configure_file(
        ${PROJECT_SOURCE_DIR}/CMakeModules/nsis/NSIS.definitions.nsh.in
        ${CMAKE_CURRENT_BINARY_DIR}/NSIS.definitions.nsh)
endif ()
