
add_library(forge_c_api_interface INTERFACE)

target_sources(forge_c_api_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/chart.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/font.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/histogram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/surface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_field.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/window.cpp
  )

target_include_directories(forge_c_api_interface
  INTERFACE
    ${PROJECT_SOURCE_DIR}/include)
