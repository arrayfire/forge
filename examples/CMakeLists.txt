CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Forge-Examples)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

MACRO(BUILD_EXAMPLE EX_NAME EX_SRC COMPUTE_NAME FG_LIBS COMPUTE_LIBS)
    IF(${COMPUTE_NAME} STREQUAL "cuda")
        CUDA_ADD_EXECUTABLE(example_${EX_NAME}_${COMPUTE_NAME} ${EX_SRC})
    ELSE()
        ADD_EXECUTABLE(example_${EX_NAME}_${COMPUTE_NAME} ${EX_SRC})
    ENDIF()
    TARGET_LINK_LIBRARIES(example_${EX_NAME}_${COMPUTE_NAME} ${COMPUTE_LIBS} ${FG_LIBS})
    SET_TARGET_PROPERTIES(example_${EX_NAME}_${COMPUTE_NAME}
        PROPERTIES
        OUTPUT_NAME ${EX_NAME}_${COMPUTE_NAME}
        RUNTIME_OUTPUT_DIRECTORY ${DIR_NAME}
        FOLDER "Examples/${COMPUTE_NAME}")
ENDMACRO()

IF(TARGET forge)
    # Do nothing
ELSE(TARGET forge)
    FIND_PACKAGE(GLEWmx REQUIRED)
    FIND_PACKAGE(Forge REQUIRED)
    SET(X11_LIBS "")
    IF(APPLE)
        FIND_PACKAGE(X11 REQUIRED)
        INCLUDE_DIRECTORIES(${X11_INCLUDE_DIR})
    ENDIF(APPLE)
    INCLUDE_DIRECTORIES(
        ${FORGE_INCLUDE_DIRECTORIES}
        ${GLEW_INCLUDE_DIR}
        )
    SET(FORGE_LIBRARIES
        ${FORGE_LIBRARIES}
        ${GLEWmx_LIBRARY}
        ${OPENGL_gl_LIBRARY}
        ${X11_LIBS}
        )
ENDIF(TARGET forge)

FILE(GLOB CPU_EXAMPLE_SRC_FILES "cpu/*.cpp")
FOREACH(FILE ${CPU_EXAMPLE_SRC_FILES})
    GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
    GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
    GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

    IF(${FORGE_FOUND})
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} ${FORGE_LIBRARIES} "")
    ELSEIF(TARGET forge)
        BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} forge "")
    ENDIF()
ENDFOREACH()

FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
    FIND_PACKAGE(NVVM REQUIRED)
    INCLUDE_DIRECTORIES(
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_NVVM_INCLUDE_DIR}
        )
    FILE(GLOB CUDA_EXAMPLE_SRC_FILES "cuda/*.cu")
    FOREACH(FILE ${CUDA_EXAMPLE_SRC_FILES})
        GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
        GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
        GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

        IF(${FORGE_FOUND})
            BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} ${FORGE_LIBRARIES} "${CUDA_NVVM_LIBRARY};${CUDA_CUDA_LIBRARY};${CUDA_LIBRARIES}")
        ELSEIF(TARGET forge)
            BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} forge "${CUDA_NVVM_LIBRARY};${CUDA_CUDA_LIBRARY};${CUDA_LIBRARIES}")
        ENDIF()
    ENDFOREACH()
ENDIF()

FIND_PACKAGE(OpenCL)
IF(OpenCL_FOUND)
    FILE(GLOB OpenCL_EXAMPLE_SRC_FILES "opencl/*.cpp")
    INCLUDE_DIRECTORIES(
        "${CMAKE_CURRENT_SOURCE_DIR}/opencl"
        ${OpenCL_INCLUDE_DIRS}
        )
    FOREACH(FILE ${OpenCL_EXAMPLE_SRC_FILES})
        GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
        GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
        GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)

        IF(${FORGE_FOUND})
            BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} ${FORGE_LIBRARIES} ${OpenCL_LIBRARIES})
        ELSEIF(TARGET forge)
            BUILD_EXAMPLE(${EXAMPLE} ${FILE} ${DIR_NAME} forge ${OpenCL_LIBRARIES})
        ENDIF()
    ENDFOREACH()
ENDIF()

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DESTINATION "${FG_INSTALL_EXAMPLE_DIR}"
    COMPONENT Examples)