# Uses SDL2_ROOT_DIR variable to look up headers
# and libraries along with standard system paths. This variable
# is quite helpful on windows platform to specify SDL2 installation
# path.
# Up on finding required files, the following variables
# will be set
# SDL2_FOUND
# SDL2_INCLUDE_DIR
# SLD2_LIBRARY

FIND_PATH(SDL2_INCLUDE_DIR SDL.h
    HINTS
    $ENV{SDL2_ROOT_DIR}
    PATH_SUFFIXES
    SDL2
    # path suffixes to search inside ENV{SDL2_ROOT_DIR}
    include/SDL include
    )

FIND_LIBRARY(SDL2_LIBRARY
    NAMES SDL2
    HINTS
    $ENV{SDL2_ROOT_DIR}
    PATH_SUFFIXES lib/x64 release debug
    PATHS
    /usr/lib
    /usr/lib64
    /usr/lib/x86_64-linux-gnu
    /usr/lib/arm-linux-gnueabihf
    /usr/local/lib
    /usr/local/lib64
    /sw/lib
    /opt/local/lib
    ${SDL2_ROOT_DIR}/lib-msvc100
    ${SDL2_ROOT_DIR}/lib-msvc110
    ${SDL2_ROOT_DIR}/lib-msvc120
    ${SDL2_ROOT_DIR}/lib
    )

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SDL2 REQUIRED_VARS SDL2_LIBRARY SDL2_INCLUDE_DIR)
MARK_AS_ADVANCED(SDL2_INCLUDE_DIR SDL2_LIBRARY)
