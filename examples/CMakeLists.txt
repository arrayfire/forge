cmake_minimum_required(VERSION 3.5)

project(Forge-Examples LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

find_package(OpenCL 1.2 QUIET)
find_package(CUDA   QUIET)
find_package(OpenGL REQUIRED)
find_package(Forge  REQUIRED)

mark_as_advanced(
    CUDA_HOST_COMPILER
    CUDA_USE_STATIC_CUDA_RUNTIME
    CUDA_rt_LIBRARY
    CMAKE_CUDA_HOST_COMPILER)

if(APPLE)
    find_package(X11 REQUIRED)
    if(X11_FOUND AND NOT TARGET X11::x11)
        add_library(X11::x11 UNKNOWN IMPORTED)
        set_target_properties(X11::x11 PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGE "C"
            IMPORTED_LOCATION "${X11_X11_LIB}"
            INTEFACE_INCLUDE_DIRECTORIES "${X11_INCLUDE_DIR}")
    endif()
endif()

include(CMakeParseArguments)
include(CMakeDependentOption)
include(InternalUtils)

option(FG_BUILD_CUDA_EXAMPLES "Turn off/on building cuda examples" ${CUDA_FOUND})
option(FG_BUILD_OPENCL_EXAMPLES "Turn off/on building opencl examples" ${OpenCL_FOUND})

cmake_dependent_option(FG_USE_SYSTEM_CL2HPP "Use system cl2.hpp header" OFF "OpenCL_FOUND" OFF)

include_if_not(FG_USE_SYSTEM_CL2HPP build_cl2hpp)

mark_as_advanced(FG_BUILD_CUDA_EXAMPLES FG_BUILD_OPENCL_EXAMPLES)

fg_deprecate(BUILD_EXAMPLES_CUDA FG_BUILD_CUDA_EXAMPLES)
fg_deprecate(BUILD_EXAMPLES_OPENCL FG_BUILD_OPENCL_EXAMPLES)
fg_deprecate(USE_SYSTEM_CL2HPP FG_USE_SYSTEM_CL2HPP)

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
            $<$<PLATFORM_ID:Darwin>:X11::x11>)
endfunction()

add_subdirectory(cpu)
conditional_directory(FG_BUILD_CUDA_EXAMPLES cuda)
conditional_directory(FG_BUILD_OPENCL_EXAMPLESCL opencl)
