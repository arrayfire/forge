

add_library(forge "")

set_target_properties(forge
  PROPERTIES
    CXX_STANDARD 11)

target_sources(forge
  PRIVATE
    chart.hpp
    chart_common.hpp
    chart_renderables.hpp
    cmap.hpp
    defines.hpp
    err_common.cpp
    err_common.hpp
    font.hpp
    handle.cpp
    handle.hpp
    image.hpp
    util.cpp
    util.hpp
    version.hpp
    window.hpp

    opengl/backend.hpp
    opengl/chart_impl.cpp
    opengl/chart_impl.hpp
    opengl/colormap_impl.cpp
    opengl/colormap_impl.hpp
    opengl/common.cpp
    opengl/common.hpp
    opengl/err_opengl.cpp
    opengl/err_opengl.hpp
    opengl/font_atlas_impl.cpp
    opengl/font_atlas_impl.hpp
    opengl/font_impl.cpp
    opengl/font_impl.hpp
    opengl/gl_native_handles.cpp
    opengl/gl_native_handles.hpp
    opengl/histogram_impl.cpp
    opengl/histogram_impl.hpp
    opengl/image_impl.cpp
    opengl/image_impl.hpp
    opengl/plot_impl.cpp
    opengl/plot_impl.hpp
    opengl/surface_impl.cpp
    opengl/surface_impl.hpp
    opengl/util.cpp
    opengl/vector_field_impl.cpp
    opengl/vector_field_impl.hpp
    opengl/window_impl.cpp
    opengl/window_impl.hpp
  )

target_include_directories(forge
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/api/c
    ${PROJECT_SOURCE_DIR}/src/backend
    ${PROJECT_SOURCE_DIR}/src/backend/opengl
    ${X11_INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}
    ${FONTCONFIG_INCLUDE_DIR}
    ${FREETYPE_INCLUDE_DIRS}
  )

if(USE_WINDOW_TOOLKIT STREQUAL "glfw3")
  target_sources(forge
    PRIVATE
      opengl/glfw/window.cpp
      opengl/glfw/window.hpp
    )
  target_include_directories(forge
    PRIVATE
      ${GLFW_INCLUDE_DIR})

  target_link_libraries(forge
    PRIVATE
      ${GLFW_LIBRARY}
    )

  target_compile_definitions(forge
    PRIVATE USE_GLFW)
elseif(USE_WINDOW_TOOLKIT STREQUAL "sdl2")
  target_sources(forge
    PRIVATE
      opengl/sdl/window.cpp
      opengl/sdl/window.hpp
    )

  target_include_directories(forge
    PRIVATE
      ${SDL2_INCLUDE_DIR})

  target_link_libraries(forge
    PRIVATE
      ${SDL2_LIBRARY}
    )
  target_compile_definitions(forge
    PRIVATE USE_SDL)
endif()

target_link_libraries(forge
  PUBLIC
    OpenGL::GL
  PRIVATE
    forge_c_api_interface
    forge_cpp_api_interface
    FreeImage::FreeImage
    glbinding::glbinding
    FontConfig::FontConfig
    #${FONTCONFIG_LIBRARY}
    ${FREETYPE_LIBRARIES}
    ${X11_LIBRARIES}
  )

set_target_properties(forge PROPERTIES
                      VERSION "${FG_VERSION}"
                      SOVERSION "${FG_VERSION_MAJOR}")

add_dependencies(forge ${glsl_shader_targets})
