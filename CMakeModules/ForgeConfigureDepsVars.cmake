# Copyright (c) 2021, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

option(FG_BUILD_OFFLINE "Build Forge assuming there is no network" OFF)

# Override fetch content base dir before including AFfetch_content
set(FETCHCONTENT_BASE_DIR "${Forge_BINARY_DIR}/extern" CACHE PATH
    "Base directory where Forge dependencies are downloaded and/or built" FORCE)

include(ForgeFetchContent)

macro(set_and_mark_depname var name)
  string(TOLOWER ${name} ${var})
  string(TOUPPER ${name} ${var}_ucname)
  mark_as_advanced(
      FETCHCONTENT_SOURCE_DIR_${${var}_ucname}
      FETCHCONTENT_UPDATES_DISCONNECTED_${${var}_ucname}
  )
endmacro()

mark_as_advanced(
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_QUIET
  FETCHCONTENT_FULLY_DISCONNECTED
  FETCHCONTENT_UPDATES_DISCONNECTED
)

set_and_mark_depname(glad_prefix "fg_glad")
set_and_mark_depname(glm_prefix "fg_glm")

if(FG_BUILD_OFFLINE)
  macro(set_fetchcontent_src_dir prefix_var dep_name)
    set(FETCHCONTENT_SOURCE_DIR_${${prefix_var}_ucname}
        "${FETCHCONTENT_BASE_DIR}/${${prefix_var}}-src" CACHE PATH
        "Source directory for ${dep_name} dependency")
    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_${${prefix_var}_ucname})
  endmacro()

  set_fetchcontent_src_dir(assets_prefix "glad")
  set_fetchcontent_src_dir(testdata_prefix "glm")
endif()
