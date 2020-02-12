add_library(forge_glad_interface INTERFACE)

target_include_directories(forge_glad_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

target_sources(forge_glad_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/glad.c>
    )

target_link_libraries(forge_glad_interface
    INTERFACE
    ${CMAKE_DL_LIBS}
    )

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS forge_glad_interface EXPORT ForgeTargets)
endif ()
