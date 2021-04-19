find_package(glfw3)
mark_as_advanced(glfw3_DIR)

add_library(forge_wtk OBJECT
    ${CMAKE_CURRENT_LIST_DIR}/window.hpp
    ${CMAKE_CURRENT_LIST_DIR}/window.cpp
    )

fg_set_target_compilation_props(forge_wtk)

target_include_directories(forge_wtk
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    PRIVATE
    ${Forge_SOURCE_DIR}/src/backend/opengl # backend's native headers
    )
