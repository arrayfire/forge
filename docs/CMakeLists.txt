# Documentation
if(NOT DEFINED FG_INSTALL_DOC_DIR)
    set(FG_INSTALL_DOC_DIR "${FG_INSTALL_DATA_DIR}/doc" CACHE PATH "Installation path for documentation")
endif()

set(FG_DOCS_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/doxygen.mk")
set(FG_DOCS_CONFIG_OUT "${CMAKE_CURRENT_BINARY_DIR}/doxygen.mk.out")

set(FG_DOCS_LAYOUT "${CMAKE_CURRENT_SOURCE_DIR}/layout.xml")
set(FG_DOCS_LAYOUT_OUT "${CMAKE_CURRENT_BINARY_DIR}/layout.xml.out")

set(DOCS_DIR     "${CMAKE_CURRENT_SOURCE_DIR}")
set(ASSETS_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/images")
set(INCLUDE_DIR  "${PROJECT_SOURCE_DIR}/include")
set(EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")
#Re-enable test snippets directory setting when doc examples are added
#SET(SNIPPETS_DIR "${PROJECT_SOURCE_DIR}/test")
configure_file(${FG_DOCS_CONFIG} ${FG_DOCS_CONFIG_OUT})
configure_file(${FG_DOCS_LAYOUT} ${FG_DOCS_LAYOUT_OUT})

###########################################################
## This generates a list of the examples cpp files and
## creates a dox file under docs/details/examples.dox
## This is used to generate documentation for examples
###########################################################
file(GLOB EXAMPLES_CPP
    "${EXAMPLES_DIR}/cpu/*.cpp"
    "${EXAMPLES_DIR}/cuda/*.cu"
    "${EXAMPLES_DIR}/opencl/*.cpp"
    "${EXAMPLES_DIR}/opencl/*.h"
    )

# Sort alphabetically
# Note: example directories will be major sort order
list(SORT EXAMPLES_CPP)

# Get filenames and write to a string
foreach(SRC ${EXAMPLES_CPP})
    get_filename_component(DIR_PATH ${SRC} DIRECTORY)
    get_filename_component(DIR_NAME ${DIR_PATH} NAME)
    get_filename_component(SRC_NAME ${SRC} NAME)
    set(EXAMPLES_LIST "${EXAMPLES_LIST}\\example ${DIR_NAME}/${SRC_NAME}\n")
endforeach(SRC ${EXAMPLES_CPP})

# Write string containing file names to examples.dox
configure_file(
    ${PROJECT_SOURCE_DIR}/CMakeModules/examples.dox.in
    ${DOCS_DIR}/details/examples.dox
)
###########################################################

add_custom_target(docs
    ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${FG_DOCS_CONFIG_OUT}
    COMMAND cmake -E copy_directory ${ASSETS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/html
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Documentation"
    VERBATIM)

# Install Doxygen documentation
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
    DESTINATION ${FG_INSTALL_DOC_DIR}
    COMPONENT documentation
    PATTERN "*"
    PATTERN ".git" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "man" EXCLUDE
    )
