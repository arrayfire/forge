#
# Make a version file that includes the ArrayFire GFX version and git revision
#
SET(AFGFX_VERSION "3.0")
SET(AFGFX_VERSION_MINOR ".beta")
EXECUTE_PROCESS(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

CONFIGURE_FILE(
    ${CMAKE_MODULE_PATH}/version.h.in
    ${CMAKE_SOURCE_DIR}/include/afgfx/version.h
)
