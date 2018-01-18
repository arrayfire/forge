
include(ExternalProject)

set(LIB_POSTFIX "")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(LIB_POSTFIX "d")
endif()

set(prefix ${PROJECT_BINARY_DIR}/third_party/ft)
set(ft_filename ${CMAKE_STATIC_LIBRARY_PREFIX}freetype${LIB_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
SET(freetype_location ${prefix}/lib/${ft_filename})

ExternalProject_Add(
    ft-ext
    GIT_REPOSITORY https://github.com/arrayfire/freetype2.git
    GIT_TAG VER-2-7-1
    PREFIX "${prefix}"
    INSTALL_DIR "${prefix}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -Wno-dev "-G${CMAKE_GENERATOR}" <SOURCE_DIR>
    -DWITH_HarfBuzz=OFF
    -DWITH_ZLIB=OFF
    -DWITH_BZip2=OFF
    -DWITH_PNG=OFF
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} -w -fPIC"
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} -w -fPIC"
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    BUILD_BYPRODUCTS ${freetype_location}
    )

ExternalProject_Get_Property(ft-ext install_dir)

set(freetype_INCLUDE_DIR ${install_dir}/include/freetype2 CACHE INTERNAL "" FORCE)
set(freetype_LIBRARY ${freetype_location} CACHE INTERNAL "" FORCE)

mark_as_advanced(freetype_INCLUDE_DIR freetype_LIBRARY)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(freetype REQUIRED_VARS freetype_LIBRARY freetype_INCLUDE_DIR)
    
if (freetype_FOUND AND NOT TARGET freetype::freetype)
    file(MAKE_DIRECTORY ${freetype_INCLUDE_DIR})
    add_library(freetype::freetype STATIC IMPORTED)
    set_target_properties(freetype::freetype PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGE "C"
        IMPORTED_LOCATION "${freetype_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${freetype_INCLUDE_DIR}")
    add_dependencies(freetype::freetype ft-ext)
endif ()
