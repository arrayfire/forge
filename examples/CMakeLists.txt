cmake_minimum_required(VERSION 3.5)

project(Forge-Examples LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

if(USE_HUNTER)
    hunter_add_package(OpenCL)
endif()

find_package(OpenGL REQUIRED)
find_package(CUDA QUIET)
find_package(OpenCL 1.2 QUIET)

if(APPLE)
    find_package(X11)
    if(X11_FOUND AND NOT TARGET X11::x11)
        add_library(X11::x11 UNKNOWN IMPORTED)
        set_target_properties(X11::x11 PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGE "C"
            IMPORTED_LOCATION "${X11_X11_LIB}"
            INTEFACE_INCLUDE_DIRECTORIES "${X11_INCLUDE_DIR}")
    endif()
endif()

option(BUILD_EXAMPLES_CUDA "Turn off/on building cuda examples" ${CUDA_FOUND})
option(BUILD_EXAMPLES_OPENCL "Turn off/on building opencl examples" ${OpenCL_FOUND})
mark_as_advanced(BUILD_EXAMPLES_CUDA BUILD_EXAMPLES_OPENCL)

include(CMakeParseArguments)
include(ConditionalDirectory)
include(CMakeDependentOption)
include(IfNotThisThenInclude)

cmake_dependent_option(USE_SYSTEM_CL2HPP "Use system cl2.hpp header" OFF "OpenCL_FOUND" OFF)

include_if_not(USE_SYSTEM_CL2HPP build_cl2hpp)

add_library(OSCompileFlags INTERFACE)

if (WIN32)
    target_compile_definitions(OSCompileFlags
        INTERFACE OS_WIN WIN32_MEAN_AND_LEAN)
elseif (APPLE)
    target_compile_definitions(OSCompileFlags INTERFACE OS_MAC)
else(WIN32)
    target_compile_definitions(OSCompileFlags INTERFACE OS_LNX)
endif(WIN32)

if(TARGET forge)
    add_library(Forge::forge ALIAS forge)
else()
    include(${Forge_BINARY_DIR}/ForgeConfig.cmake)
endif()

function(add_example target_name source backend)
    set(options CXX11)
    set(single_value_args "")
    set(multiple_value_args INCLUDE_DIRS LIBRARIES)
    cmake_parse_arguments(arg "${options}" "${single_value_args}" "${multiple_value_args}" ${ARGN})

    set(target "example_${target_name}_${backend}")

    string(TOLOWER ${backend} lowerCaseBackend)

    if (${lowerCaseBackend} STREQUAL "cuda" AND ${CMAKE_VERSION} VERSION_LESS "3.10.0")
        cuda_add_executable(${target} ${source})
    else ()
        add_executable(${target} ${source})
    endif ()

    set_target_properties(${target} PROPERTIES OUTPUT_NAME "${target_name}")
    set_target_properties(${target} PROPERTIES FOLDER Examples/${backend})

    if (${arg_CXX11})
        set_target_properties(${target} PROPERTIES CXX_STANDARD 11)
    endif (${arg_CXX11})

    target_include_directories(${target} PRIVATE ${arg_INCLUDE_DIRS})

    target_link_libraries(${target}
            OSCompileFlags
            OpenGL::GL
            Forge::forge
            ${arg_LIBRARIES}
            $<$<PLATFORM_ID:Darwin>:X11::x11>
        )
endfunction()

add_subdirectory(cpu)
conditional_directory(BUILD_EXAMPLES_CUDA cuda)
conditional_directory(BUILD_EXAMPLES_OPENCL opencl)
