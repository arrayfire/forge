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

add_custom_target(docs
    ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${FG_DOCS_CONFIG_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Documentation"
    VERBATIM)

set(SPHINX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx")
set(SPHINX_INDEX_FILE ${SPHINX_BUILD_DIR}/index.html)
set(DOXY_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/xml)
set(DOXY_INDEX_FILE ${DOXY_OUT_DIR}/index.xml)

file(GLOB restFiles rst/*.rst)
file(GLOB mdFiles markdown/*.md)

# Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
    COMMAND
    ${SPHINX_EXECUTABLE} -Dbreathe_projects.Forge=${DOXY_OUT_DIR}
    ${SPHINX_SOURCE_DIR} ${SPHINX_BUILD_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
    ${DOXY_INDEX_FILE}
    ${restFiles}
    ${mdFiles}
    MAIN_DEPENDENCY ${SPHINX_SOURCE_DIR}/conf.py
    COMMENT "Generating documentation with Sphinx")

# Nice named target so we can run the job easily
add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE} docs)

# Install Doxygen documentation
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sphinx
    DESTINATION ${FG_INSTALL_DOC_DIR}
    COMPONENT forge_dev
    PATTERN "*"
    PATTERN ".git" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "man" EXCLUDE
    )
