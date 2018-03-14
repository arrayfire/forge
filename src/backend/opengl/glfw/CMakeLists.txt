if(USE_HUNTER)
    hunter_add_package(glfw)
endif()

find_package(glfw3)

add_library(wtk_interface INTERFACE)

target_include_directories(wtk_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/backend/opengl>
    )

target_sources(wtk_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/window.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/window.cpp>
    )

target_link_libraries(wtk_interface INTERFACE glfw)

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS wtk_interface EXPORT ForgeTargets)
endif ()
