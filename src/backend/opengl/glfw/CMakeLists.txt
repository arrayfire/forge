if(FG_ENABLE_HUNTER)
    hunter_add_package(glfw)
    find_package(glfw3 NO_CMAKE_PATH NO_SYSTEM_ENVIRONMENT_PATH)
else ()
    find_package(glfw3)
endif()

add_library(forge_wtk_interface INTERFACE)

target_include_directories(forge_wtk_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/backend/opengl>
    )

target_sources(forge_wtk_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/window.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/window.cpp>
    )

target_link_libraries(forge_wtk_interface INTERFACE glfw)

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS forge_wtk_interface EXPORT ForgeTargets)
endif ()

if (WIN32)
    install(FILES $<TARGET_FILE:glfw>
    DESTINATION ${FG_INSTALL_BIN_DIR}
    COMPONENT dependencies)
endif ()
