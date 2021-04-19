add_library(forge_common_obj_lib OBJECT
    ${CMAKE_CURRENT_LIST_DIR}/chart_common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/chart.hpp
    ${CMAKE_CURRENT_LIST_DIR}/chart_renderables.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cmap.hpp
    ${CMAKE_CURRENT_LIST_DIR}/defines.hpp
    ${CMAKE_CURRENT_LIST_DIR}/err_handling.hpp
    ${CMAKE_CURRENT_LIST_DIR}/err_handling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/font.hpp
    ${CMAKE_CURRENT_LIST_DIR}/handle.hpp
    ${CMAKE_CURRENT_LIST_DIR}/handle.cpp
    ${CMAKE_CURRENT_LIST_DIR}/image.hpp
    ${CMAKE_CURRENT_LIST_DIR}/util.hpp
    ${CMAKE_CURRENT_LIST_DIR}/util.cpp
    ${CMAKE_CURRENT_LIST_DIR}/window.hpp
    )

fg_set_target_compilation_props(forge_common_obj_lib)

target_include_directories(forge_common_obj_lib
    PRIVATE
    ${Forge_SOURCE_DIR}/src/backend/opengl # For backend native headers
    $<TARGET_PROPERTY:forge_wtk,INTERFACE_INCLUDE_DIRECTORIES>
    )
