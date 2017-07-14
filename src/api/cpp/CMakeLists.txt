
add_library(forge_cpp_api_interface INTERFACE)

target_sources(forge_cpp_api_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/chart.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/error.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/font.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/histogram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/surface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_field.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/window.cpp
  )

target_include_directories(forge_cpp_api_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(forge_cpp_api_interface
    INTERFACE forge_c_api_interface)
