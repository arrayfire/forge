# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/nsis")

include(Version)
include(CPackIFW)

set(VENDOR_NAME "ArrayFire")
set(APP_NAME "Forge")
set(APP_URL "www.arrayfire.com")

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

  set(LICENSE_FILE       "${Forge_SOURCE_DIR}/LICENSE")
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
    set(CPACK_RESOURCE_FILE_LICENSE "${Forge_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_README "${Forge_SOURCE_DIR}/README.md")
endif()

# Set the default components installed in the package
get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)

include(CPackComponent)

cpack_add_install_type(Development
  DISPLAY_NAME "Development")
cpack_add_install_type(Extra
  DISPLAY_NAME "Extra")
cpack_add_install_type(Runtime
  DISPLAY_NAME "Runtime")

cpack_add_component_group(backends
  DISPLAY_NAME "Forge"
  DESCRIPTION "Forge libraries")

cpack_add_component(dependencies
  DISPLAY_NAME "Forge Dependencies"
  DESCRIPTION "Libraries required by Forge OpenGL backend"
  PARENT_GROUP backends
  INSTALL_TYPES Development Runtime)

cpack_add_component(forge
  DISPLAY_NAME "Forge"
  DESCRIPTION "Forge library."
  PARENT_GROUP backends
  DEPENDS dependencies
  INSTALL_TYPES Development Runtime)

cpack_add_component(documentation
  DISPLAY_NAME "Documentation"
  DESCRIPTION "Forge documentation files"
  INSTALL_TYPES Extra)

cpack_add_component(headers
  DISPLAY_NAME "C/C++ Headers"
  DESCRIPTION "Development headers for the Forge library."
  INSTALL_TYPES Development)

cpack_add_component(cmake
  DISPLAY_NAME "CMake Support"
  DESCRIPTION "Configuration files to use ArrayFire using CMake."
  INSTALL_TYPES Development)

cpack_add_component(examples
  DISPLAY_NAME "Forge Examples"
  DESCRIPTION "Various examples using Forge."
  INSTALL_TYPES Extra)

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
cpack_ifw_configure_component(dependencies)
cpack_ifw_configure_component(forge)
cpack_ifw_configure_component(documentation)
cpack_ifw_configure_component(headers)
cpack_ifw_configure_component(cmake)
cpack_ifw_configure_component(examples)

##
# Debian package
##
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEB_COMPONENT_INSTALL ON)
#set(CMAKE_INSTALL_RPATH /usr/lib;${Forge_BUILD_DIR}/third_party/forge/lib)
#set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE http://www.arrayfire.com)

##
# RPM package
##
set(CPACK_RPM_PACKAGE_LICENSE "BSD")
set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")

set(CPACK_PACKAGE_GROUP "Development/Libraries")
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
