find_package(OpenGL REQUIRED)

set(BackendTarget "forge")

add_library(${BackendTarget} "")

set_target_properties(${BackendTarget} PROPERTIES FOLDER Forge)
set_target_properties(${BackendTarget} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${BackendTarget} PROPERTIES CXX_STANDARD 11)
set_target_properties(${BackendTarget} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${BackendTarget} PROPERTIES VERSION ${Forge_VERSION})
set_target_properties(${BackendTarget} PROPERTIES SOVERSION ${Forge_VERSION_MAJOR})

if (WIN32)
    target_compile_definitions(${BackendTarget}
        PUBLIC FGDLL
        PRIVATE OS_WIN WIN32_MEAN_AND_LEAN)

    # C4251: Warnings about dll interfaces. Thrown by glbinding, may be fixed in
    #        the future
    # C4068: Warnings about unknown pragmas
    # C4275: Warnings about using non-exported classes as base class of an
    #        exported class
    set_target_properties(${BackendTarget} PROPERTIES COMPILE_FLAGS "/wd4251 /wd4068 /wd4275")
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
endif()

if(WITH_FREEIMAGE)
    target_compile_definitions(${BackendTarget} PRIVATE USE_FREEIMAGE)
    if (USE_STATIC_FREEIMAGE)
        target_link_libraries(${BackendTarget} PRIVATE FreeImage::FreeImage_STATIC)
    else ()
        target_link_libraries(${BackendTarget} PRIVATE FreeImage::FreeImage)
    endif ()
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
endif()

#--------------------------------------------------------------------
# Install forget target
#--------------------------------------------------------------------
install(TARGETS forge
    EXPORT ForgeTargets
    COMPONENT forge
    PUBLIC_HEADER DESTINATION fg
    RUNTIME DESTINATION ${FG_INSTALL_BIN_DIR}
    LIBRARY DESTINATION ${FG_INSTALL_LIB_DIR}
    ARCHIVE DESTINATION ${FG_INSTALL_LIB_DIR}
    FRAMEWORK DESTINATION framework
    INCLUDES DESTINATION ${FG_INSTALL_INC_DIR}
    )

# install dependencies
get_target_library_path(binary_location forge)
if (EXISTS "${binary_location}")
    get_prerequisites(${binary_location} dependencies 1 1 "" "${prerequisites_search_dirs}")
    resolve_dependencies_paths(forge_deps
        "${dependencies}" "${binary_location}" "${prerequisites_search_dirs}")
    install(FILES ${forge_deps}
        DESTINATION ${FG_INSTALL_BIN_DIR}
        COMPONENT   dependencies)
endif ()
