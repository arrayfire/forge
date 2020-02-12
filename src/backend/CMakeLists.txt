# BkndTargetName is the target used by
# rendering backend(opengl/vulkan) to add
# source files and set target properties
string(TOLOWER ${PROJECT_NAME} BkndTargetName)

# Until we move up minimum cmake version to 3.14
# add_library and target_link_libraryies have to be
# at same level or one level above in cmakelists-file-tree
add_library(${BkndTargetName} "")

add_dependencies(${BkndTargetName} ${glsl_shader_targets})

if(${FG_RENDERING_BACKEND} STREQUAL "OpenGL")
    find_package(OpenGL REQUIRED)
    add_subdirectory(opengl)
    target_include_directories(${BkndTargetName} PUBLIC ${OPENGL_INCLUDE_DIR})
    target_link_libraries(${BkndTargetName} PUBLIC ${OPENGL_gl_LIBRARY})
elseif(${FG_RENDERING_BACKEND} STREQUAL "Vulkan") # arm is not possible now
    message(FATAL_ERROR "Vulkan rendering support not added yet")
endif()

set_target_properties(${BkndTargetName}
    PROPERTIES
    FOLDER ${PROJECT_NAME}
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_INLINES_HIDDEN YES
    LINKER_LANGUAGE CXX
    VERSION ${Forge_VERSION}
    SOVERSION ${Forge_VERSION_MAJOR}
    )

if(WIN32)
    target_compile_definitions(${BkndTargetName}
        PRIVATE FGDLL OS_WIN WIN32_MEAN_AND_LEAN)

    # C4068: Warnings about unknown pragmas
    # C4275: Warnings about using non-exported classes as base class of an
    #        exported class
    set_target_properties(${BkndTargetName} PROPERTIES COMPILE_FLAGS "/wd4068 /wd4275")
elseif(APPLE)
    target_compile_definitions(${BkndTargetName} PRIVATE OS_MAC)
else(WIN32)
    target_compile_definitions(${BkndTargetName} PRIVATE OS_LNX)
endif(WIN32)

target_include_directories(${BkndTargetName}
    PUBLIC
    $<BUILD_INTERFACE:${Forge_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${Forge_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${FG_INSTALL_INC_DIR}>
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    SYSTEM PRIVATE
    ${FREETYPE_INCLUDE_DIRS}
    ${Forge_SOURCE_DIR}/extern/glm
    )

if(FG_WITH_FREEIMAGE)
    target_compile_definitions(${BkndTargetName} PRIVATE USE_FREEIMAGE)
    if(FG_USE_STATIC_FREEIMAGE)
        target_link_libraries(${BkndTargetName} PRIVATE FreeImage::FreeImage_STATIC)
    else()
        target_link_libraries(${BkndTargetName} PRIVATE FreeImage::FreeImage)
    endif()
endif()

target_link_libraries(${BkndTargetName}
    PRIVATE
    Boost::boost
    forge_backend_interface
    forge_c_api_interface
    forge_cpp_api_interface
    forge_glad_interface
    forge_wtk_interface
    )

if(FG_ENABLE_HUNTER)
    target_link_libraries(${BkndTargetName} PRIVATE freetype::freetype)
else()
    target_link_libraries(${BkndTargetName} PRIVATE ${FREETYPE_LIBRARIES})
endif()

if(UNIX)
    target_link_libraries(${BkndTargetName}
        PRIVATE
        FontConfig::FontConfig
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<BOOL:${FG_USE_STATIC_CPPFLAGS}>>:-static-libstdc++>
        $<$<AND:$<C_COMPILER_ID:GNU>,$<BOOL:${FG_USE_STATIC_CPPFLAGS}>>:-static-libgcc>
        )
endif(UNIX)

source_group(include
    REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/include/*)
source_group("api\\cpp"
    REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/api/cpp/*)
source_group("api\\c"
    REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/api/c/*)
source_group("backend"
    REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/backend/common/*)
source_group("backend\\shaders"
    REGULAR_EXPRESSION ${Forge_SOURCE_DIR}/src/backend/glsl_shaders/*)

#--------------------------------------------------------------------
# Install forge target
#--------------------------------------------------------------------
install(TARGETS ${BkndTargetName}
    EXPORT ForgeTargets
    COMPONENT forge
    PUBLIC_HEADER DESTINATION fg
    RUNTIME DESTINATION ${FG_INSTALL_BIN_DIR}
    LIBRARY DESTINATION ${FG_INSTALL_LIB_DIR}
    ARCHIVE DESTINATION ${FG_INSTALL_LIB_DIR}
    FRAMEWORK DESTINATION framework
    INCLUDES DESTINATION ${FG_INSTALL_INC_DIR})

# install dependencies
if(WIN32 AND BUILD_SHARED_LIBS AND
        FG_WITH_FREEIMAGE AND NOT FG_USE_STATIC_FREEIMAGE)
    install(FILES $<TARGET_FILE:FreeImage::FreeImage>
        DESTINATION ${FG_INSTALL_BIN_DIR}
        COMPONENT dependencies)
endif()
