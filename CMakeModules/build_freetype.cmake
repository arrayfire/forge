INCLUDE(ExternalProject)

SET(prefix ${CMAKE_BINARY_DIR}/third_party/freetype)

ExternalProject_Add(
    freetype-external
    GIT_REPOSITORY git://git.sv.nongnu.org/freetype/freetype2.git
    GIT_TAG 14df6b1a63f5c5773bb498063205cb79aac21173
    PREFIX "${prefix}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -Wno-dev "-G${CMAKE_GENERATOR}" ${prefix}/src/freetype-external
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} -w -fPIC"
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} -w -fPIC"
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:STRING=${prefix}
    )

ADD_LIBRARY(freetype IMPORTED STATIC)
SET(freetype_location ${prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}freetype${CMAKE_STATIC_LIBRARY_SUFFIX})
SET_TARGET_PROPERTIES(freetype PROPERTIES IMPORTED_LOCATION ${freetype_location})
ADD_DEPENDENCIES(freetype freetype-external)

SET(FREETYPE_INCLUDE_DIRS ${prefix}/include)
SET(FREETYPE_LIBRARIES freetype)
SET(FREETYPE_FOUND ON)
