add_library(forge_backend_interface INTERFACE)

# non private sources that are used for build only in INTERFACE target types
# should make sense only for builds, not when installed. Hence, wrapped with
# $<BUILD_INTERFACE:...> generator expression. Without it, cmake generate step
# fails when building static library.
target_sources(forge_backend_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/chart_common.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/chart.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/chart_renderables.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cmap.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/defines.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/err_handling.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/err_handling.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/font.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/handle.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/handle.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/image.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/util.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/util.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/window.hpp>
    )

target_include_directories(forge_backend_interface
    INTERFACE
    $<BUILD_INTERFACE:${Forge_SOURCE_DIR}/include>
    )

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS forge_backend_interface EXPORT ForgeTargets)
endif ()
