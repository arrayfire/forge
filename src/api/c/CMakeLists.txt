add_library(c_api_interface INTERFACE)

target_sources(c_api_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/chart.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/exception.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/font.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/histogram.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/image.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/plot.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/surface.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vector_field.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/window.cpp>
    )

target_include_directories(c_api_interface
    INTERFACE
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:backend_interface,INTERFACE_INCLUDE_DIRECTORIES>>
    )

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS c_api_interface EXPORT ForgeTargets)
endif ()
