# Copyright (c) 2021, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

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

function(conditional_directory variable directory)
    if(${variable})
        add_subdirectory(${directory})
    endif()
endfunction()
