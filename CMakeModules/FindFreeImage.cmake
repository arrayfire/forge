# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
#
# Targets defined by this script
#   FreeImage::FreeImage
#   FreeImage::FreeImage_STATIC
#
# Note:
# 1. The static version target is only defined if the static lib is found
# 2. Environment variable FreeImage_ROOT can be defined on Windows where
#    FreeImage is just a zip file of header and library files.
#
# Sets the following variables:
#          FreeImage_FOUND
#          FreeImage_INCLUDE_DIR
#          FreeImage_LINK_LIBRARY
#          FreeImage_STATIC_LIBRARY
#          FreeImage_DLL_LIBRARY - Windows only
#
# Usage:
# find_package(FreeImage)
# if (FreeImage_FOUND)
#    target_link_libraries(mylib PRIVATE FreeImage::FreeImage)
# endif (FreeImage_FOUND)
#
# OR if you want to link against the static library:
#
# find_package(FreeImage)
# if (FreeImage_FOUND)
#    target_link_libraries(mylib PRIVATE FreeImage::FreeImage_STATIC)
# endif (FreeImage_FOUND)
#
# NOTE: You do not need to include the FreeImage include directories since they
# will be included as part of the target_link_libraries command

find_path(FreeImage_INCLUDE_DIR
  NAMES FreeImage.h
  PATHS
    /usr
    /usr/local
    /sw
    /opt/local
    ${FreeImage_ROOT}
  PATH_SUFFIXES
    include
  DOC "The directory where FreeImage.h resides")

find_library(FreeImage_LINK_LIBRARY
  NAMES FreeImage freeimage
  PATHS
    /usr
    /usr/local
    /sw
    /opt/local
    ${FreeImage_ROOT}
  PATH_SUFFIXES
    lib
    lib64
    x64/lib
  DOC "The FreeImage library")

find_library(FreeImage_STATIC_LIBRARY
  NAMES
    ${CMAKE_STATIC_LIBRARY_PREFIX}FreeImageLIB${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${CMAKE_STATIC_LIBRARY_PREFIX}FreeImage${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${CMAKE_STATIC_LIBRARY_PREFIX}freeimage${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATHS
    /usr
    /usr/local
    /sw
    /opt/local
    ${FreeImage_ROOT}
  PATH_SUFFIXES
    lib
    lib64
    x64/lib
  DOC "The FreeImage static library")

if (WIN32)
  find_file(FreeImage_DLL_LIBRARY
    NAMES
      ${CMAKE_SHARED_LIBRARY_PREFIX}FreeImage${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${CMAKE_SHARED_LIBRARY_PREFIX}freeimage${CMAKE_SHARED_LIBRARY_SUFFIX}
    PATHS
      ${FreeImage_ROOT}
    PATH_SUFFIXES
      lib
      lib64
      x64/lib
      bin
      x64/bin
    DOC "The FreeImage dll")
  mark_as_advanced(FreeImage_DLL_LIBRARY)
endif ()

mark_as_advanced(
  FreeImage_INCLUDE_DIR
  FreeImage_LINK_LIBRARY
  FreeImage_STATIC_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FreeImage
  REQUIRED_VARS FreeImage_INCLUDE_DIR FreeImage_LINK_LIBRARY)

if(FreeImage_FOUND AND NOT TARGET FreeImage::FreeImage)
  add_library(FreeImage::FreeImage SHARED IMPORTED)
  if(WIN32)
    set_target_properties(FreeImage::FreeImage
      PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGE "C"
        INTERFACE_INCLUDE_DIRECTORIES "${FreeImage_INCLUDE_DIR}"
        IMPORTED_LOCATION "${FreeImage_DLL_LIBRARY}"
        IMPORTED_IMPLIB "${FreeImage_LINK_LIBRARY}")
  else(WIN32)
    set_target_properties(FreeImage::FreeImage
      PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGE "C"
        INTERFACE_INCLUDE_DIRECTORIES "${FreeImage_INCLUDE_DIR}"
        IMPORTED_LOCATION "${FreeImage_LINK_LIBRARY}"
        IMPORTED_NO_SONAME FALSE)
  endif(WIN32)
endif()

if(FreeImage_STATIC_LIBRARY AND NOT TARGET FreeImage::FreeImage_STATIC)
  add_library(FreeImage::FreeImage_STATIC STATIC IMPORTED)
  set_target_properties(FreeImage::FreeImage_STATIC
    PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      INTERFACE_INCLUDE_DIRECTORIES "${FreeImage_INCLUDE_DIR}"
      IMPORTED_LOCATION "${FreeImage_STATIC_LIBRARY}")
endif()
