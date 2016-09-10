INCLUDE(ExternalProject)

SET(prefix ${PROJECT_BINARY_DIR}/third_party/ft)
SET(freetype_location ${prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}freetype${CMAKE_STATIC_LIBRARY_SUFFIX})
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
    GIT_TAG 14df6b1a63f5c5773bb498063205cb79aac21173
    PREFIX "${prefix}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -Wno-dev "-G${CMAKE_GENERATOR}" <SOURCE_DIR>
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${CXXFLAGS}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${CFLAGS}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:STRING=${prefix}
    ${byproducts}
    )

ADD_LIBRARY(freetype IMPORTED STATIC)
SET(freetype_location ${prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}freetype${CMAKE_STATIC_LIBRARY_SUFFIX})
SET_TARGET_PROPERTIES(freetype PROPERTIES IMPORTED_LOCATION ${freetype_location})
ADD_DEPENDENCIES(freetype ft-ext)

SET(FREETYPE_INCLUDE_DIRS ${prefix}/include/freetype2)
SET(FREETYPE_LIBRARIES freetype)
SET(FREETYPE_FOUND ON)
