find_package(OpenGL REQUIRED)

set(BackendTarget "forge")

add_library(${BackendTarget} SHARED "")

set_target_properties(${BackendTarget} PROPERTIES FOLDER Forge)
set_target_properties(${BackendTarget} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${BackendTarget} PROPERTIES CXX_STANDARD 11)
set_target_properties(${BackendTarget} PROPERTIES LINKER_LANGUAGE CXX)

if (WIN32)
    target_compile_definitions(${BackendTarget}
        PUBLIC FGDLL
        PRIVATE OS_WIN WIN32_MEAN_AND_LEAN)
elseif (APPLE)
    target_compile_definitions(${BackendTarget} PRIVATE OS_MAC)
else(WIN32)
    target_compile_definitions(${BackendTarget} PRIVATE OS_LNX)
endif(WIN32)

target_sources(${BackendTarget}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backend.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/chart_impl.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/chart_impl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/colormap_impl.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/colormap_impl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/err_opengl.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/err_opengl.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/surface_impl.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/surface_impl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/update_buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vector_field_impl.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vector_field_impl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/window_impl.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/window_impl.cpp
    )

if(${WINDOW_TOOLKIT} STREQUAL "glfw3")
    add_subdirectory(glfw)
    target_compile_definitions(${BackendTarget} PRIVATE USE_GLFW)
elseif(${WINDOW_TOOLKIT} STREQUAL "sdl2")
    add_subdirectory(sdl)
    target_compile_definitions(${BackendTarget} PRIVATE USE_SDL)
elseif(${WINDOW_TOOLKIT} STREQUAL "Qt5")
    find_package(Qt5Core REQUIRED) #need it for the AUTOMOC to work on the target
    add_subdirectory(qt5)
    target_compile_definitions(${BackendTarget} PRIVATE USE_QT5)
    set_target_properties(${BackendTarget} PROPERTIES AUTOMOC TRUE)
endif()

if(WITH_FREEIMAGE)
    target_compile_definitions(${BackendTarget} PRIVATE USE_FREEIMAGE)
    target_link_libraries(${BackendTarget} PRIVATE FreeImage::FreeImage)
endif()

target_include_directories(${BackendTarget}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${X11_INCLUDE_DIR}
    )

target_link_libraries(${BackendTarget}
    PUBLIC
        OpenGL::GL
    PRIVATE
        glbinding::glbinding
        glm
        freetype::freetype
        Boost::boost
        backend_interface
        wtk_interface
        c_api_interface
        cpp_api_interface
    )

if(UNIX)
    target_link_libraries(${BackendTarget}
        PRIVATE
          FontConfig::FontConfig
          ${X11_X11_LIB}
	)
endif(UNIX)

add_dependencies(${BackendTarget} ${glsl_shader_targets})

source_group(include            REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/include/*)
source_group("api\\cpp"         REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/api/cpp/*)
source_group("api\\c"           REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/api/c/*)
source_group("backend"          REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/backend/common/*|${CMAKE_CURRENT_SOURCE_DIR}/*)
source_group("backend\\shaders" REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/backend/glsl_shaders/*)

if(${WINDOW_TOOLKIT} STREQUAL "glfw3")
    source_group("backend\\glfw" REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/backend/opengl/glfw/*)
elseif(${WINDOW_TOOLKIT} STREQUAL "sdl2")
    source_group("backend\\sdl2" REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/backend/opengl/sdl/*)
elseif(${WINDOW_TOOLKIT} STREQUAL "Qt5")
    source_group("backend\\qt5" REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/backend/opengl/qt5/*)
endif()

#--------------------------------------------------------------------
# Install forget target
#--------------------------------------------------------------------
install(TARGETS forge
    EXPORT ForgeConfig
    ARCHIVE DESTINATION ${FG_INSTALL_LIB_DIR}
    LIBRARY DESTINATION ${FG_INSTALL_LIB_DIR}
    RUNTIME DESTINATION ${FG_INSTALL_BIN_DIR}
    )

install(EXPORT ForgeConfig NAMESPACE Forge:: DESTINATION ${FG_INSTALL_CMAKE_DIR})

#Following ForgeConfig.cmake is built using local build tree
#Don't copy this file anywhere else
export(TARGETS forge NAMESPACE Forge:: FILE ForgeConfig.cmake)
