find_package(glfw3)

add_library(forge_wtk_interface INTERFACE)

target_include_directories(forge_wtk_interface
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    )

target_sources(forge_wtk_interface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/window.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/window.cpp>
    )

target_link_libraries(forge_wtk_interface INTERFACE glfw)

if(NOT BUILD_SHARED_LIBS)
    install(TARGETS forge_wtk_interface EXPORT ForgeTargets)
endif()

if(WIN32 AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_FILE:glfw>
    DESTINATION ${FG_INSTALL_BIN_DIR}
    COMPONENT dependencies)
endif()

mark_as_advanced(glfw3_DIR)
