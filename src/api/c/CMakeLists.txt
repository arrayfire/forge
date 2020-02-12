add_library(forge_c_api_interface INTERFACE)

# non private sources that are used for build only in INTERFACE target types
# should make sense only for builds, not when installed. Hence, wrapped with
# $<BUILD_INTERFACE:...> generator expression. Without it, cmake generate step
# fails when building static library.
target_sources(forge_c_api_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/chart.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/exception.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/font.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/histogram.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/image.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/plot.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/surface.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/vector_field.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/window.cpp>
    )

target_include_directories(forge_c_api_interface
    INTERFACE
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:forge_backend_interface,INTERFACE_INCLUDE_DIRECTORIES>>
    )

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS forge_c_api_interface EXPORT ForgeTargets)
endif ()
