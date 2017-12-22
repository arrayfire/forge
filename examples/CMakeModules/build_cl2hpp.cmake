if(NOT TARGET cl2hpp-ext)
    set(FILE_URL "https://github.com/KhronosGroup/OpenCL-CLHPP/releases/download/v2.0.10/cl2.hpp")
    set(CL_HEADER "${PROJECT_BINARY_DIR}/third_party/cl2hpp/include/CL/cl2.hpp")
    file(DOWNLOAD ${FILE_URL} ${CL_HEADER})
    get_filename_component(DOWNLOAD_DIR ${CL_HEADER} DIRECTORY)
endif()

message(STATUS "Found cl2.hpp header at: ${DOWNLOAD_DIR}")

if (NOT TARGET OpenCL::cl2hpp)
    add_library(OpenCL::cl2hpp IMPORTED INTERFACE GLOBAL)
    add_dependencies(OpenCL::cl2hpp cl2hpp-ext)

    set_target_properties(OpenCL::cl2hpp PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${DOWNLOAD_DIR}/..")
endif ()
