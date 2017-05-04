INCLUDE(ExternalProject)

SET(prefix ${PROJECT_BINARY_DIR}/third_party/ft)

SET(LIB_POSTFIX "")
IF (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    SET(LIB_POSTFIX "d")
ENDIF()

SET(freetype_location
    ${prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}freetype${LIB_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})

IF(CMAKE_VERSION VERSION_LESS 3.2)
    IF(CMAKE_GENERATOR MATCHES "Ninja")
        MESSAGE(WARNING "Building freetype with Ninja has known issues with CMake older than 3.2")
    endif()
    SET(byproducts)
ELSE()
    SET(byproducts BYPRODUCTS ${freetype_location})
ENDIF()


IF(UNIX)
    SET(CXXFLAGS "${CMAKE_CXX_FLAGS} -w -fPIC")
    SET(CFLAGS "${CMAKE_C_FLAGS} -w -fPIC")
ENDIF(UNIX)

ExternalProject_Add(
    ft-ext
    GIT_REPOSITORY http://git.sv.nongnu.org/r/freetype/freetype2.git
    GIT_TAG VER-2-7-1
    PREFIX "${prefix}"
    INSTALL_DIR "${prefix}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -Wno-dev "-G${CMAKE_GENERATOR}" <SOURCE_DIR>
    -DWITH_HarfBuzz:BOOL=OFF
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${CXXFLAGS}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${CFLAGS}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    ${byproducts}
    )

ADD_LIBRARY(freetype IMPORTED STATIC)

ExternalProject_Get_Property(ft-ext install_dir)

SET_TARGET_PROPERTIES(freetype PROPERTIES IMPORTED_LOCATION ${freetype_location})

ADD_DEPENDENCIES(freetype ft-ext)

# Based on the instructions on freetype site
# for header path inclusion given at below URL
# https://www.freetype.org/freetype2/docs/tutorial/step1.html#section-1
# we are including <root>/include/freetype2 as FREETYPE_INCLUDE_DIRS
SET(FREETYPE_INCLUDE_DIRS "${install_dir}/include/freetype2" CACHE INTERNAL "" FORCE)
SET(FREETYPE_LIBRARIES ${freetype_location} CACHE INTERNAL "" FORCE)
SET(FREETYPE_FOUND ON CACHE INTERNAL "" FORCE)
