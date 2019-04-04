find_package(OpenGL REQUIRED)

target_sources(${BkndTargetName}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/abstract_renderable.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/chart_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/chart_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colormap_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colormap_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gl_helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gl_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/font_atlas_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/font_atlas_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/font_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/font_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gl_native_handles.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gl_native_handles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/histogram_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/histogram_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plot_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plot_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shader_program.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shader_program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/surface_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/surface_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/update_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_field_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_field_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/window_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/window_impl.cpp
  )

if(${FG_USE_WINDOW_TOOLKIT} STREQUAL "glfw3")
  add_subdirectory(glfw)
  target_compile_definitions(${BkndTargetName} PRIVATE USE_GLFW)
elseif(${FG_USE_WINDOW_TOOLKIT} STREQUAL "sdl2")
  add_subdirectory(sdl)
  target_compile_definitions(${BkndTargetName} PRIVATE USE_SDL)
endif()

target_include_directories(${BkndTargetName}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

target_link_libraries(${BkndTargetName}
  PUBLIC
    OpenGL::GL
  PRIVATE
    forge_wtk_interface
  )

source_group("backend"
  REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(${FG_USE_WINDOW_TOOLKIT} STREQUAL "glfw3")
  source_group("backend\\glfw"
    REGULAR_EXPRESSION ${PROJECT_SOURCE_DIR}/src/backend/opengl/glfw/*)
elseif(${FG_USE_WINDOW_TOOLKIT} STREQUAL "sdl2")
  source_group("backend\\sdl2"
    REGULAR_EXPRESSION ${PROJECT_SOURCE_DIR}/src/backend/opengl/sdl/*)
endif()
