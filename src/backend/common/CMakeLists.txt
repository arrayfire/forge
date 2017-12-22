add_library(backend_interface INTERFACE)

target_sources(backend_interface
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/chart_common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/chart.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/chart_renderables.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cmap.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/defines.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/err_common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/err_common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/font.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/handle.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/handle.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/util.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/window.hpp
    )

target_include_directories(backend_interface
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
    )
