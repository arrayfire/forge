set(cl2hpp_header
    "${PROJECT_BINARY_DIR}/third_party/cl2hpp/include/CL/cl2.hpp")
if (OpenCL_FOUND)
    if (NOT EXISTS ${cl2hpp_header})
        set(file_url
            "https://github.com/KhronosGroup/OpenCL-CLHPP/releases/download/v2.0.10/cl2.hpp")
        file(DOWNLOAD ${file_url} ${cl2hpp_header}
            EXPECTED_HASH MD5=c38d1b78cd98cc809fa2a49dbd1734a5
            STATUS download_result)
        list(GET download_result 0 download_code)
        if (NOT ${download_code} EQUAL 0)
            file(REMOVE ${cl2hpp_header}) #empty file have to be removed
            message(FATAL_ERROR "Failed to download cl2hpp header")
        endif ()
    endif ()
    get_filename_component(download_dir ${cl2hpp_header} DIRECTORY)
    message(STATUS "Found cl2.hpp header at: ${download_dir}")
    if (NOT TARGET OpenCL::cl2hpp)
        add_library(OpenCL::cl2hpp IMPORTED INTERFACE GLOBAL)
        set_target_properties(OpenCL::cl2hpp PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${download_dir}/..")
    endif ()
endif ()
