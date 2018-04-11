# Author: Pradeep Garigipati <pradeep@arrayfire.com>
#
# Heavy work is done by FindFreetype.cmake that comes along with
# FreeType installation on your system if your OS is OSX/Unix.
#
# If the OS is Windows, the freetype is build as external project
# using the script build_freetype.cmake
#
# Sets the following variables:
#          FreeType_FOUND
#          FreeType_INCLUDE_DIR
#          FreeType_LIBRARY
#
# Usage:
# find_package(FreeType)
# if (FreeType_FOUND)
#    target_link_libraries(mylib PRIVATE freetype::freetype)
# endif (FreeType_FOUND)
#
# OR if you want to link against the static library:
#
# find_package(FreeType)
# if (FreeType_FOUND)
#    target_link_libraries(mylib PRIVATE freetype::freetype_STATIC)
# endif (FreeType_FOUND)
#
# NOTE: You do not need to include the Freetype include directories since they
# will be included as part of the target_link_libraries command
if(WIN32)
    include(build_freetype)
else(WIN32)
    find_package(Freetype REQUIRED)
    set(FreeType_INCLUDE_DIR ${FREETYPE_INCLUDE_DIRS})
    set(FreeType_LIBRARY ${FREETYPE_LIBRARIES})

    mark_as_advanced(
        FreeType_INCLUDE_DIR
        FreeType_LIBRARY
        )

    include(FindPackageHandleStandardArgs)

    find_package_handle_standard_args(FreeType
        REQUIRED_VARS FreeType_LIBRARY FreeType_INCLUDE_DIR
        )

    if (FreeType_FOUND AND NOT TARGET freetype::freetype)
        add_library(freetype::freetype UNKNOWN IMPORTED)
        set_target_properties(freetype::freetype PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGE "C"
            IMPORTED_LOCATION "${FreeType_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FreeType_INCLUDE_DIR}")
    endif ()
endif(WIN32)
