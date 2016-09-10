FIND_PACKAGE(Doxygen REQUIRED)

# Documentation
IF(NOT DEFINED FG_INSTALL_DOC_DIR)
    SET(FG_INSTALL_DOC_DIR "${FG_INSTALL_DATA_DIR}/doc" CACHE PATH "Installation path for documentation")
ENDIF()

SET(FG_DOCS_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/doxygen.mk")
SET(FG_DOCS_CONFIG_OUT "${CMAKE_CURRENT_BINARY_DIR}/doxygen.mk.out")

SET(FG_DOCS_LAYOUT "${CMAKE_CURRENT_SOURCE_DIR}/layout.xml")
SET(FG_DOCS_LAYOUT_OUT "${CMAKE_CURRENT_BINARY_DIR}/layout.xml.out")

SET(DOCS_DIR     "${CMAKE_CURRENT_SOURCE_DIR}")
SET(ASSETS_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/images")
SET(INCLUDE_DIR  "${PROJECT_SOURCE_DIR}/include")
SET(EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")
#Re-enable test snippets directory setting when doc examples are added
#SET(SNIPPETS_DIR "${PROJECT_SOURCE_DIR}/test")
CONFIGURE_FILE(${FG_DOCS_CONFIG} ${FG_DOCS_CONFIG_OUT})
CONFIGURE_FILE(${FG_DOCS_LAYOUT} ${FG_DOCS_LAYOUT_OUT})

###########################################################
## This generates a list of the examples cpp files and
## creates a dox file under docs/details/examples.dox
## This is used to generate documentation for examples
###########################################################
FILE(GLOB EXAMPLES_CPP
    "${EXAMPLES_DIR}/cpu/*.cpp"
    "${EXAMPLES_DIR}/cuda/*.cu"
    "${EXAMPLES_DIR}/opencl/*.cpp"
    "${EXAMPLES_DIR}/opencl/*.h"
    )

# Sort alphabetically
# Note: example directories will be major sort order
LIST(SORT EXAMPLES_CPP)

# Get filenames and write to a string
FOREACH(SRC ${EXAMPLES_CPP})
    GET_FILENAME_COMPONENT(DIR_PATH ${SRC} DIRECTORY)
    GET_FILENAME_COMPONENT(DIR_NAME ${DIR_PATH} NAME)
    GET_FILENAME_COMPONENT(SRC_NAME ${SRC} NAME)
    SET(EXAMPLES_LIST "${EXAMPLES_LIST}\\example ${DIR_NAME}/${SRC_NAME}\n")
ENDFOREACH(SRC ${EXAMPLES_CPP})

# Write string containing file names to examples.dox
CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/CMakeModules/examples.dox.in
    ${DOCS_DIR}/details/examples.dox
)
###########################################################

ADD_CUSTOM_TARGET(docs
    ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${FG_DOCS_CONFIG_OUT}
    COMMAND cmake -E copy_directory ${ASSETS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/html
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Documentation"
    VERBATIM)

# Install Doxygen documentation
INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
    DESTINATION ${FG_INSTALL_DOC_DIR}
    COMPONENT documentation
    PATTERN "*"
    PATTERN ".git" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "man" EXCLUDE
    )
