CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

OPTION(USE_SYSTEM_GLBINDING "Use system glbinding" OFF)
IF(USE_SYSTEM_GLBINDING)
    # Point glbinding_DIR (case sensitive) to the location of glbinding-config.cmake
    # This file is generally at CMAKE_INSTALL_PREFIX/glbinding-config.cmake of
    # the glbinding project
    FIND_PACKAGE(glbinding REQUIRED)
ELSE(USE_SYSTEM_GLBINDING)
    INCLUDE(build_glbinding)
ENDIF(USE_SYSTEM_GLBINDING)

INCLUDE(GLSLtoH)

OPTION(USE_SYSTEM_GLM "Use the system GLM library" OFF)
IF(USE_SYSTEM_GLM)
  FIND_PACKAGE(glm REQUIRED)
ELSE()
  INCLUDE(build_glm)
ENDIF()

OPTION(USE_SYSTEM_FREETYPE "Use the system Freetype library" OFF)
IF(USE_SYSTEM_FREETYPE)
  FIND_PACKAGE(Freetype REQUIRED)
ELSE()
  INCLUDE(build_freetype)
ENDIF()

IF(UNIX)
    FIND_PACKAGE(FontConfig REQUIRED)
ENDIF(UNIX)

IF(${USE_WINDOW_TOOLKIT} STREQUAL "glfw3")
    FIND_PACKAGE(GLFW REQUIRED)
    IF(GLFW_FOUND)
        SET(WTK_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})
        SET(WTK_LIBRARIES ${GLFW_LIBRARY})
        ADD_DEFINITIONS(-DUSE_GLFW)
    ELSE(GLFW_FOUND)
        MESSAGE(FATAL_ERROR "GLFW not found")
    ENDIF(GLFW_FOUND)
ENDIF()

IF(${USE_WINDOW_TOOLKIT} STREQUAL "sdl2")
    FIND_PACKAGE(SDL2 REQUIRED)
    IF(SDL2_FOUND)
        SET(WTK_INCLUDE_DIRS ${SDL2_INCLUDE_DIR})
        SET(WTK_LIBRARIES ${SDL2_LIBRARY})
        ADD_DEFINITIONS(-DUSE_SDL)
    ELSE(SDL2_FOUND)
        MESSAGE(FATAL_ERROR "SDL2 not found")
    ENDIF(SDL2_FOUND)
ENDIF()


IF(WIN32)
    ADD_DEFINITIONS(-DFGDLL)
ENDIF(WIN32)

# OS Definitions
IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ELSE(UNIX) #Windows
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF(UNIX)

# DON'T REMOVE CURRENT BINARY DIRECTORY FROM
# INCLUDE_DIRECTORIES, IT IS USED TO FIND
# AUTO-GENERATED GLSL HEADERS
INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${FREETYPE_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
    ${WTK_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    "${PROJECT_SOURCE_DIR}/src/api/c"
    "${PROJECT_SOURCE_DIR}/src/api/cpp"
    "${PROJECT_SOURCE_DIR}/src/backend"
    )

IF(UNIX)
    INCLUDE_DIRECTORIES("${FONTCONFIG_INCLUDE_DIR}")
ENDIF(UNIX)

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

SET(glsl_shader_headers
    "shader_headers")
FILE(GLOB glsl_shaders
    "shaders/*.glsl")

LIST(SORT c_api_headers)
LIST(SORT c_api_sources)
LIST(SORT cpp_api_headers)
LIST(SORT cpp_api_sources)
LIST(SORT bknd_headers)
LIST(SORT bknd_sources)
LIST(SORT gl_headers)
LIST(SORT gl_sources)
LIST(SORT glsl_shaders)

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

SOURCE_GROUP("src\\backend\\opengl\\glsl" FILES ${glsl_shaders})

GLSL_TO_H(
    SOURCES ${glsl_shaders}
    VARNAME  shader_files
    EXTENSION "hpp"
    OUTPUT_DIR ${glsl_shader_headers}
    TARGETS glsl_shader_targets
    NAMESPACE "glsl"
    EOD "0"
    )

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
IF(USE_SYSTEM_GLBINDING)
    TARGET_LINK_LIBRARIES(forge PRIVATE glbinding::glbinding)
ELSE(USE_SYSTEM_GLBINDING)
    INCLUDE_DIRECTORIES(${GLBINDING_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(forge PRIVATE ${GLBINDING_LIBRARIES})
    ADD_DEPENDENCIES(forge glbinding)
ENDIF(USE_SYSTEM_GLBINDING)

SET_TARGET_PROPERTIES(forge PROPERTIES
                      VERSION ${FG_VERSION}
                      SOVERSION 0)

INSTALL(TARGETS forge
        EXPORT FORGE
        DESTINATION "${FG_INSTALL_LIB_DIR}"
        COMPONENT libraries)

EXPORT(TARGETS forge FILE ForgeTargets.cmake)

INSTALL(EXPORT FORGE DESTINATION "${FG_INSTALL_CMAKE_DIR}"
        COMPONENT cmake
        FILE ForgeTargets.cmake
       )
