#
# Sets Forge installation paths.
#

# NOTE: These paths are all relative to the project installation prefix.

# Executables
if(NOT DEFINED FG_INSTALL_BIN_DIR)
    set(FG_INSTALL_BIN_DIR "bin" CACHE PATH "Installation path for executables")
endif()

# Libraries
if(NOT DEFINED FG_INSTALL_LIB_DIR)
    set(FG_INSTALL_LIB_DIR "lib" CACHE PATH "Installation path for libraries")
endif()

# Header files
if(NOT DEFINED FG_INSTALL_INC_DIR)
    set(FG_INSTALL_INC_DIR "include" CACHE PATH "Installation path for headers")
endif()

# Data files
if(NOT DEFINED FG_INSTALL_DATA_DIR)
    set(FG_INSTALL_DATA_DIR "share/Forge" CACHE PATH "Installation path for data files")
endif()

# Documentation
if(NOT DEFINED FG_INSTALL_DOC_DIR)
    set(FG_INSTALL_DOC_DIR "${FG_INSTALL_DATA_DIR}/doc" CACHE PATH "Installation path for documentation")
endif()

if(NOT DEFINED FG_INSTALL_EXAMPLE_DIR)
    set(FG_INSTALL_EXAMPLE_DIR "${FG_INSTALL_DATA_DIR}" CACHE PATH "Installation path for examples")
endif()

# Man pages
if(NOT DEFINED FG_INSTALL_MAN_DIR)
    set(FG_INSTALL_MAN_DIR "${FG_INSTALL_DATA_DIR}/man" CACHE PATH "Installation path for man pages")
endif()

# CMake files
if(NOT DEFINED FG_INSTALL_CMAKE_DIR)
    set(FG_INSTALL_CMAKE_DIR "${FG_INSTALL_DATA_DIR}/cmake" CACHE PATH "Installation path for CMake files")
endif()
