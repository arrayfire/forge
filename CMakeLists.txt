CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(FORGE)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
INCLUDE(FGInstallDirs)

SET(USE_WINDOW_TOOLKIT "glfw3" CACHE STRING "Choose Window toolkit")
SET_PROPERTY(CACHE USE_WINDOW_TOOLKIT PROPERTY STRINGS "glfw3" "sdl2")

OPTION(BUILD_DOCUMENTATION "Build Documentation" OFF)
OPTION(BUILD_EXAMPLES "Build Examples" ON)

OPTION(USE_LOCAL_GLM "Download and use local GLM" OFF)
OPTION(USE_LOCAL_FREETYPE "Download and use local freetype" OFF)
MARK_AS_ADVANCED(USE_LOCAL_GLM)
MARK_AS_ADVANCED(USE_LOCAL_FREETYPE)

# Set a default build type if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
      "MinSizeRel" "RelWithDebInfo")
ENDIF()

IF(UNIX)
    ADD_DEFINITIONS(-Wall -std=c++11 -fvisibility=hidden)
ENDIF()

# OS Definitions
IF(UNIX)
    IF(APPLE)
        ADD_DEFINITIONS(-DOS_MAC)
        SET(CMAKE_MACOSX_RPATH ON)
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

        LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}" isSystemDir)
        IF("${isSystemDir}" STREQUAL "-1")
            SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
        ENDIF("${isSystemDir}" STREQUAL "-1")
    ELSE(APPLE)
        ADD_DEFINITIONS(-DOS_LNX)
    ENDIF(APPLE)
ELSE(UNIX)
    ADD_DEFINITIONS(-DOS_WIN -DNOMINMAX)
ENDIF(UNIX)

INCLUDE(${CMAKE_MODULE_PATH}/Version.cmake)

FIND_PACKAGE(FreeImage REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)

INCLUDE_DIRECTORIES(
    "${PROJECT_SOURCE_DIR}/include"
    ${OPENGL_INCLUDE_DIR}
    ${FREEIMAGE_INCLUDE_PATH}
    )

SET(X11_LIBS "")
IF(APPLE)
    FIND_PACKAGE(X11 REQUIRED)
    INCLUDE_DIRECTORIES(${X11_INCLUDE_DIR})
    SET(X11_LIBS ${X11_LIBRARIES})
ENDIF(APPLE)

ADD_EXECUTABLE(glsl2cpp "${CMAKE_MODULE_PATH}/glsl2cpp.cpp")

ADD_SUBDIRECTORY(src/backend/opengl)

IF(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF()

# Generate documentation
IF(BUILD_DOCUMENTATION)
    ADD_SUBDIRECTORY(docs)
ENDIF(BUILD_DOCUMENTATION)

#--------------------------------------------------------------------
# Create generated files
#--------------------------------------------------------------------
INCLUDE(CMakePackageConfigHelpers)

configure_package_config_file("${CMAKE_MODULE_PATH}/ForgeConfig.cmake.in"
                              "${CMAKE_CURRENT_BINARY_DIR}/ForgeConfig.cmake"
                              INSTALL_DESTINATION ${FG_INSTALL_CMAKE_DIR}
                              PATH_VARS CMAKE_INSTALL_PREFIX
                              NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ForgeConfigVersion.cmake"
                                 VERSION ${FG_VERSION}
                                 COMPATIBILITY SameMajorVersion)

#--------------------------------------------------------------------
# Install files other than the library, examples and docs
# The library is installed by src/CMakeLists.txt
#--------------------------------------------------------------------
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION "${FG_INSTALL_INC_DIR}"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE
)

INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/ForgeConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/ForgeConfigVersion.cmake"
              DESTINATION ${FG_INSTALL_CMAKE_DIR})

# We are only installing the source files for examples.
# So this should not depend on the value of BUILD_EXAMPLES
# These examples will be installed without building the executables for the examples
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"
        DESTINATION "${FG_INSTALL_EXAMPLE_DIR}"
        COMPONENT examples)

INSTALL(FILES
        "${CMAKE_MODULE_PATH}/FindOpenCL.cmake"
        "${CMAKE_MODULE_PATH}/FindGLEWmx.cmake"
        DESTINATION "${FG_INSTALL_EXAMPLE_DIR}/CMakeModules/"
        COMPONENT examples)
