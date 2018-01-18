if(USE_HUNTER)
    hunter_add_package(SDL2)
endif()

find_package(SDL2)

add_library(wtk_interface INTERFACE)

target_include_directories(wtk_interface
    INTERFACE
        ${CMAKE_SOURCE_DIR}/src/backend/opengl
    )

target_sources(wtk_interface
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/window.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/window.cpp
    )

target_link_libraries(wtk_interface INTERFACE SDL2::SDL2)
