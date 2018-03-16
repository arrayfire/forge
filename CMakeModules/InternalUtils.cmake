# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

function(dependency_check VAR ERROR_MESSAGE)
    if(NOT ${VAR})
        message(SEND_ERROR ${ERROR_MESSAGE})
    endif()
endfunction()

# Includes the directory if the variable is set
function(conditional_directory variable directory)
    if(${variable})
        add_subdirectory(${directory})
    endif()
endfunction()

function(get_native_path out_path path)
    file(TO_NATIVE_PATH ${path} native_path)
    string(REPLACE "\\" "\\\\" native_path  ${native_path})
    set(${out_path} ${native_path} PARENT_SCOPE)
endfunction()

function(resolve_dependencies_paths out_deps in_deps context search_dirs)
    set(out_list "")
    foreach(current_dependency ${in_deps})
        gp_resolve_item(${context} "${current_dependency}" "" "${search_dirs}" resolved_file)
        list(APPEND out_list "${resolved_file}")
    endforeach()
    set(${out_deps} ${out_list} PARENT_SCOPE)
endfunction()

function(__fg_deprecate_var var access value)
  if(access STREQUAL "READ_ACCESS")
      message(DEPRECATION "Variable ${var} is deprecated. Use FG_${var} instead.")
  endif()
endfunction()

function(fg_deprecate var newvar)
  if(DEFINED ${var})
    message(DEPRECATION "Variable ${var} is deprecated. Use ${newvar} instead.")
    get_property(doc CACHE ${newvar} PROPERTY HELPSTRING)
    set(${newvar} ${${var}} CACHE BOOL "${doc}" FORCE)
    unset(${var} CACHE)
  endif()
  variable_watch(${var} __fg_deprecate_var)
endfunction()

# mark CUDA cmake cache variables as advanced
# this should have been taken care of by FindCUDA I think.
mark_as_advanced(
    CMAKE_CUDA_HOST_COMPILER
    CUDA_HOST_COMPILER
    CUDA_SDK_ROOT_DIR
    CUDA_TOOLKIT_ROOT_DIR
    CUDA_USE_STATIC_CUDA_RUNTIME
    CUDA_rt_LIBRARY
    )
