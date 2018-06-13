#
# Make a version file that includes the Forge version and git revision
#
set(FG_VERSION_MAJOR ${Forge_VERSION_MAJOR})
set(FG_VERSION_MINOR ${Forge_VERSION_MINOR})
set(FG_VERSION_PATCH ${Forge_VERSION_PATCH})

set(FG_VERSION "${FG_VERSION_MAJOR}.${FG_VERSION_MINOR}.${FG_VERSION_PATCH}")
set(FG_API_VERSION_CURRENT ${FG_VERSION_MAJOR}${FG_VERSION_MINOR})

# From CMake 3.0.0 CMAKE_<LANG>_COMPILER_ID is AppleClang for OSX machines
# that use clang for compilations
if("${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang")
    set(COMPILER_NAME "AppleClang")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(COMPILER_NAME "LLVM Clang")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(COMPILER_NAME "GNU Compiler Collection(GCC/G++)")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    set(COMPILER_NAME "Intel Compiler")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(COMPILER_NAME "Microsoft Visual Studio")
endif()

set(COMPILER_VERSION "${CMAKE_C_COMPILER_VERSION}")
set(FG_COMPILER_STRING "${COMPILER_NAME} ${COMPILER_VERSION}")

execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT GIT_COMMIT_HASH)
    message(STATUS "No git. Setting hash to default")
    set(GIT_COMMIT_HASH "default")
endif()

configure_file(
    ${PROJECT_SOURCE_DIR}/CMakeModules/version.h.in
    ${PROJECT_BINARY_DIR}/include/fg/version.h)

configure_file(
    ${PROJECT_SOURCE_DIR}/CMakeModules/version.hpp.in
    ${PROJECT_BINARY_DIR}/src/backend/common/version.hpp)
