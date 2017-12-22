CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

FIND_PACKAGE(OpenGL REQUIRED)

INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})

OPTION(USE_SYSTEM_GLBINDING "Use system glbinding" OFF)
IF(USE_SYSTEM_GLBINDING)
    # Point glbinding_DIR (case sensitive) to the location of glbinding-config.cmake
    # This file is generally at CMAKE_INSTALL_PREFIX/glbinding-config.cmake of
    # the glbinding project
    FIND_PACKAGE(glbinding REQUIRED)
ELSE(USE_SYSTEM_GLBINDING)
    INCLUDE(build_glbinding)
ENDIF(USE_SYSTEM_GLBINDING)

IF(WIN32)
    ADD_DEFINITIONS(-DFGDLL)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${WTK_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${PROJECT_SOURCE_DIR}/src/api/c"
    "${PROJECT_SOURCE_DIR}/src/api/cpp"
    "${PROJECT_SOURCE_DIR}/src/backend"
    )

FILE(GLOB api_headers
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/fg/*.h"
    )

FILE(GLOB c_api_headers
    "${PROJECT_SOURCE_DIR}/src/api/c/*.hpp"
    )
FILE(GLOB c_api_sources
    "${PROJECT_SOURCE_DIR}/src/api/c/*.cpp"
    )
FILE(GLOB cpp_api_headers
    "${PROJECT_SOURCE_DIR}/src/api/cpp/*.hpp"
    )
FILE(GLOB cpp_api_sources
    "${PROJECT_SOURCE_DIR}/src/api/cpp/*.cpp"
    )

FILE(GLOB bknd_headers
    "${PROJECT_SOURCE_DIR}/src/backend/*.hpp"
    )

FILE(GLOB bknd_sources
    "${PROJECT_SOURCE_DIR}/src/backend/*.cpp"
    )

FILE(GLOB gl_headers
    "*.hpp"
    "*.h"
    )

FILE(GLOB gl_sources
    "*.c"
    "*.cpp"
    )

LIST(SORT c_api_headers)
LIST(SORT c_api_sources)
LIST(SORT cpp_api_headers)
LIST(SORT cpp_api_sources)
LIST(SORT bknd_headers)
LIST(SORT bknd_sources)
LIST(SORT gl_headers)
LIST(SORT gl_sources)

SOURCE_GROUP(inc FILES ${api_headers})

SOURCE_GROUP("src\\api\\c\\headers" FILES ${c_api_headers})
SOURCE_GROUP("src\\api\\c\\sources" FILES ${c_api_sources})

SOURCE_GROUP("src\\api\\cpp\\headers" FILES ${cpp_api_headers})
SOURCE_GROUP("src\\api\\cpp\\sources" FILES ${cpp_api_sources})

SOURCE_GROUP("src\\backend" FILES ${bknd_headers})
SOURCE_GROUP("src\\backend" FILES ${bknd_sources})

SOURCE_GROUP("src\\backend\\opengl\\headers" FILES ${gl_headers})
SOURCE_GROUP("src\\backend\\opengl\\sources" FILES ${gl_sources})

IF(${USE_WINDOW_TOOLKIT} STREQUAL "glfw3")
    FILE(GLOB wtk_headers
        "glfw/*.hpp")
    FILE(GLOB wtk_sources
        "glfw/*.cpp")
    SOURCE_GROUP("src\\backend\\opengl\\glfw\\headers" FILES ${wtk_headers})
    SOURCE_GROUP("src\\backend\\opengl\\glfw\\sources" FILES ${wtk_sources})
ELSEIF(${USE_WINDOW_TOOLKIT} STREQUAL "sdl2")
    FILE(GLOB wtk_headers
        "sdl/*.hpp")
    FILE(GLOB wtk_sources
        "sdl/*.cpp")
    SOURCE_GROUP("src\\backend\\opengl\\sdl\\headers" FILES ${wtk_headers})
    SOURCE_GROUP("src\\backend\\opengl\\sdl\\sources" FILES ${wtk_sources})
ENDIF()

ADD_LIBRARY(forge SHARED
    ${api_headers}
    ${c_api_headers}
    ${c_api_sources}
    ${cpp_api_headers}
    ${cpp_api_sources}
    ${bknd_headers}
    ${bknd_sources}
    ${gl_headers}
    ${gl_sources}
    ${wtk_headers}
    ${wtk_sources}
    )

TARGET_LINK_LIBRARIES(forge
    PRIVATE ${FONTCONFIG_LIBRARY}
    PRIVATE ${FREETYPE_LIBRARIES}
    PRIVATE ${WTK_LIBRARIES}
    PRIVATE ${OPENGL_gl_LIBRARY}
    PRIVATE ${FREEIMAGE_LIBRARY}
    PRIVATE ${X11_LIBS}
    )

ADD_DEPENDENCIES(forge ${glsl_shader_targets})

IF (NOT USE_SYSTEM_FREETYPE AND NOT USE_HUNTER)
    ADD_DEPENDENCIES(forge freetype)
ENDIF()

IF(USE_SYSTEM_GLBINDING)
    TARGET_LINK_LIBRARIES(forge PRIVATE glbinding::glbinding)
ELSE(USE_SYSTEM_GLBINDING)
    INCLUDE_DIRECTORIES(${GLBINDING_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(forge PRIVATE glbinding)
    ADD_DEPENDENCIES(forge glbinding)
ENDIF(USE_SYSTEM_GLBINDING)

SET_TARGET_PROPERTIES(forge PROPERTIES
                      VERSION "${FG_VERSION}"
                      SOVERSION "${FG_VERSION_MAJOR}")

INSTALL(TARGETS forge
        EXPORT FORGE
        DESTINATION "${FG_INSTALL_LIB_DIR}"
        COMPONENT libraries)

EXPORT(TARGETS forge FILE ForgeTargets.cmake)

INSTALL(EXPORT FORGE DESTINATION "${FG_INSTALL_CMAKE_DIR}"
        COMPONENT cmake
        FILE ForgeTargets.cmake
       )
