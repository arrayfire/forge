# BkndTargetName is the target used by
# rendering backend(opengl/vulkan) to add
# source files and set target properties
string(TOLOWER ${PROJECT_NAME} BkndTargetName)

add_library(${BkndTargetName} "")

set_target_properties(${BkndTargetName}
  PROPERTIES
    FOLDER ${PROJECT_NAME}
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 11
    LINKER_LANGUAGE CXX
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )

if (WIN32)
  target_compile_definitions(${BkndTargetName}
    PRIVATE FGDLL OS_WIN WIN32_MEAN_AND_LEAN)

  # C4068: Warnings about unknown pragmas
  # C4275: Warnings about using non-exported classes as base class of an
  #        exported class
  set_target_properties(${BkndTargetName} PROPERTIES COMPILE_FLAGS "/wd4068 /wd4275")
elseif (APPLE)
  target_compile_definitions(${BkndTargetName} PRIVATE OS_MAC)
else(WIN32)
  target_compile_definitions(${BkndTargetName} PRIVATE OS_LNX)
endif(WIN32)

if(FG_WITH_FREEIMAGE)
  target_compile_definitions(${BkndTargetName} PRIVATE USE_FREEIMAGE)
  if (FG_USE_STATIC_FREEIMAGE)
    target_link_libraries(${BkndTargetName} PRIVATE FreeImage::FreeImage_STATIC)
  else ()
    target_link_libraries(${BkndTargetName} PRIVATE FreeImage::FreeImage)
  endif ()
endif()

target_include_directories(${BkndTargetName}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/extern/glm
  )

target_link_libraries(${BkndTargetName}
  PRIVATE
    Boost::boost
    forge_backend_interface
    forge_c_api_interface
    forge_cpp_api_interface
    forge_glad_interface
  )

if (FG_ENABLE_HUNTER)
  target_link_libraries(${BkndTargetName} PRIVATE freetype::freetype)
else ()
  target_link_libraries(${BkndTargetName} PRIVATE Freetype::Freetype)
endif ()

if(UNIX)
  target_link_libraries(${BkndTargetName}
  PRIVATE
    FontConfig::FontConfig
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<BOOL:${FG_USE_STATIC_CPPFLAGS}>>:-static-libstdc++>
    $<$<AND:$<C_COMPILER_ID:GNU>,$<BOOL:${FG_USE_STATIC_CPPFLAGS}>>:-static-libgcc>
  )
endif(UNIX)

if (WIN32)
  # C4275: Warnings about using non-exported classes as base class of an
  # exported class
  target_compile_options(${BkndTargetName} PUBLIC /wd4275)
endif ()

add_dependencies(${BkndTargetName} ${glsl_shader_targets})

source_group(include
  REGULAR_EXPRESSION ${PROJECT_SOURCE_DIR}/include/*)
source_group("api\\cpp"
  REGULAR_EXPRESSION ${PROJECT_SOURCE_DIR}/src/api/cpp/*)
source_group("api\\c"
  REGULAR_EXPRESSION ${PROJECT_SOURCE_DIR}/src/api/c/*)
source_group("backend"
  REGULAR_EXPRESSION ${PROJECT_SOURCE_DIR}/src/backend/common/*)
source_group("backend\\shaders"
  REGULAR_EXPRESSION ${PROJECT_SOURCE_DIR}/src/backend/glsl_shaders/*)

add_subdirectory(opengl)

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
if (WIN32 AND BUILD_SHARED_LIBS AND
    FG_WITH_FREEIMAGE AND NOT FG_USE_STATIC_FREEIMAGE)
  install(FILES $<TARGET_FILE:FreeImage::FreeImage>
    DESTINATION ${FG_INSTALL_BIN_DIR}
    COMPONENT dependencies)
endif ()
