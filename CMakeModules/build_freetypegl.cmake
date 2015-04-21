INCLUDE(ExternalProject)

SET(prefix ${CMAKE_BINARY_DIR}/third_party/freetype-gl)

ExternalProject_Add(
    freetype-gl-external
    GIT_REPOSITORY https://github.com/9prady9/freetype-gl.git
    GIT_TAG 859ed45435a24d32d10aeb9f62432e3f4be5988a
    PREFIX "${prefix}"
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -Wno-dev "-G${CMAKE_GENERATOR}" ${prefix}/src/freetype-gl-external
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} -w -fPIC"
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} -w -fPIC"
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -Dfreetype-gl_BUILD_DEMOS:BOOL=OFF
    )

ADD_LIBRARY(freetype-gl IMPORTED STATIC)
SET(freetype_gl_location ${prefix}/src/freetype-gl-external-build/${CMAKE_STATIC_LIBRARY_PREFIX}freetype-gl${CMAKE_STATIC_LIBRARY_SUFFIX})
SET_TARGET_PROPERTIES(freetype-gl PROPERTIES IMPORTED_LOCATION ${freetype_gl_location})
ADD_DEPENDENCIES(freetype-gl freetype-gl-external)

SET(FREETYPEGL_INCLUDE_DIRS ${prefix}/src/freetype-gl-external)
SET(FREETYPEGL_LIBRARIES freetype-gl)
