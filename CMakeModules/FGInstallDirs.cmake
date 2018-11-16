#
# Sets Forge installation paths.
#

# NOTE: These paths are all relative to the project installation prefix.

include(GNUInstallDirs)

# Executables
if(NOT DEFINED FG_INSTALL_BIN_DIR)
    set(FG_INSTALL_BIN_DIR "lib" CACHE PATH "Installation path for executables")
endif()

# Libraries
if(NOT DEFINED FG_INSTALL_LIB_DIR)
  if(WIN32)
      set(FG_INSTALL_LIB_DIR "lib" CACHE PATH "Installation path for libraries")
  else()
      set(FG_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Installation path for libraries")
  endif()
endif()

# Header files
if(NOT DEFINED FG_INSTALL_INC_DIR)
    set(FG_INSTALL_INC_DIR "include" CACHE PATH "Installation path for headers")
endif()

set(DATA_DIR "share/Forge")

# Documentation
if(NOT DEFINED FG_INSTALL_DOC_DIR)
    if (WIN32)
        set(docs_dir "doc")
    else ()
        set(docs_dir "${DATA_DIR}/doc")
    endif ()
    set(FG_INSTALL_DOC_DIR "${docs_dir}" CACHE PATH "Installation path for documentation")
endif()

if(NOT DEFINED FG_INSTALL_EXAMPLE_DIR)
    if (WIN32)
        set(examples_dir "examples")
    else ()
        set(examples_dir "${DATA_DIR}/examples")
    endif ()
    set(FG_INSTALL_EXAMPLE_DIR "${examples_dir}" CACHE PATH "Installation path for examples")
endif()

# Man pages
if(NOT DEFINED FG_INSTALL_MAN_DIR)
    set(FG_INSTALL_MAN_DIR "${DATA_DIR}/man" CACHE PATH "Installation path for man pages")
endif()

# CMake files
if(NOT DEFINED FG_INSTALL_CMAKE_DIR)
    if(WIN32)
        set(cmake_dir "cmake")
    else()
        set(cmake_dir "${DATA_DIR}/cmake")
    endif()
    set(FG_INSTALL_CMAKE_DIR "${cmake_dir}" CACHE PATH "Installation path for CMake files")
endif()

# Use absolute paths (these changes are internal and will not show up in cache)
# The cache will continue to show relative/absolute paths as used without modifications
# This is required for configure_package_config_file in CMakeLists.txt

# CMAKE_INSTALL_PREFIX
# If this is relative, it is relative to PROJECT_BINARY_DIR
if(NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
    get_filename_component(CMAKE_INSTALL_PREFIX
                          "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PREFIX}"
                           ABSOLUTE)
endif()

mark_as_advanced(
    FG_INSTALL_CMAKE_DIR
    FG_INSTALL_MAN_DIR
    FG_INSTALL_EXAMPLE_DIR
    FG_INSTALL_DOC_DIR
    FG_INSTALL_INC_DIR
    FG_INSTALL_LIB_DIR
    FG_INSTALL_BIN_DIR)
