CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(FORGE)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

OPTION(BUILD_EXAMPLES "Build Examples" ON)

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
  IF(APPLE)   #OSX
    ADD_DEFINITIONS(-DOS_MAC)
  ELSE(APPLE) #Linux
    ADD_DEFINITIONS(-DOS_LNX)
  ENDIF()
  FIND_PACKAGE(FontConfig REQUIRED)
ELSE(${UNIX}) #Windows
  ADD_DEFINITIONS(-DOS_WIN -DNOMINMAX)
ENDIF()

# Architechture Definitions
#INCLUDE(${CMAKE_MODULE_PATH}/TargetArch.cmake)
#target_architecture(ARCH)
#IF(${ARCH} STREQUAL "x86_64")
#  ADD_DEFINITIONS(-DARCH_64)
#ELSE(${ARCH})
#  ADD_DEFINITIONS(-DARCH_32)
#ENDIF()

INCLUDE(${CMAKE_MODULE_PATH}/Version.cmake)

ADD_SUBDIRECTORY(src)

## Install step
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE
)
