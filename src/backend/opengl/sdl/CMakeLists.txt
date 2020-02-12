if(FG_ENABLE_HUNTER)
    hunter_add_package(SDL2)
    find_package(SDL2 NO_CMAKE_PATH NO_SYSTEM_ENVIRONMENT_PATH)
else()
    find_package(SDL2)
endif()

add_library(forge_wtk_interface INTERFACE)

target_include_directories(forge_wtk_interface
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    )

target_sources(forge_wtk_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/window.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/window.cpp>
    )

target_link_libraries(forge_wtk_interface INTERFACE SDL2::SDL2)

if(NOT BUILD_SHARED_LIBS)
    install(TARGETS forge_wtk_interface EXPORT ForgeTargets)
endif()

if(WIN32 AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_FILE:SDL2::SDL2>
    DESTINATION ${FG_INSTALL_BIN_DIR}
    COMPONENT dependencies)
endif()
