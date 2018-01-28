# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

# Platform specific settings
#
# Add paths and flags specific platforms. This can inc

if(APPLE)
  # Some homebrew libraries(glbinding) are not installed in directories that
  # CMake searches by default.
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/usr/local/opt")
endif()

function(get_target_library_path out_path target)
  set(library_full_name "${CMAKE_SHARED_LIBRARY_PREFIX}${target}${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(binary_path "${CMAKE_CURRENT_BINARY_DIR}/${library_full_name}")
  if (WIN32)
    set(binary_path "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${library_full_name}")
  endif ()
  get_native_path(full_path ${binary_path})
  set(${out_path} ${full_path} PARENT_SCOPE)
endfunction()
