file(DOWNLOAD
  "https://github.com/arrayfire/forge/blob/master/.github/LICENSE"
  "${PROJECT_BINARY_DIR}/LICENSE.md"
  STATUS fg_check_result
  TIMEOUT 4
)
list(GET fg_check_result 0 fg_is_connected)

set(cl2hpp_extern_header
    "${Forge_SOURCE_DIR}/extern/cl2hpp/include/CL/cl2.hpp")
set(cl2hpp_download_header
    "${PROJECT_BINARY_DIR}/third_party/cl2hpp/include/CL/cl2.hpp")

if(${fg_is_connected} AND NOT EXISTS ${cl2hpp_download_header})
    if(EXISTS ${cl2hpp_extern_header})
        get_filename_component(download_dir ${cl2hpp_extern_header} DIRECTORY)
        message(STATUS "Found cl2.hpp header at: ${download_dir}")
        if (NOT TARGET OpenCL::cl2hpp)
            add_library(OpenCL::cl2hpp IMPORTED INTERFACE GLOBAL)
            set_target_properties(OpenCL::cl2hpp PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${download_dir}/..")
        endif ()
    else()
        message(FATAL_ERROR [=[
            Offline builds require already available cl2hpp header. Please
            download the file from and place it at the below path  path under
            project root directory.

            Download URL: https://github.com/KhronosGroup/OpenCL-CLHPP/releases/download/v2.0.10/cl2.hpp
            Target Location: extern/cl2hpp/include/CL
            ]=])
    endif()
else()
    # Any CMakeLists.txt file including this file should call find_package(OpenCL)
    if (OpenCL_FOUND)
        if (NOT EXISTS ${cl2hpp_download_header})
            set(file_url
                "https://github.com/KhronosGroup/OpenCL-CLHPP/releases/download/v2.0.10/cl2.hpp")
            file(DOWNLOAD ${file_url} ${cl2hpp_download_header}
                EXPECTED_HASH MD5=c38d1b78cd98cc809fa2a49dbd1734a5
                STATUS download_result)
            list(GET download_result 0 download_code)
            if (NOT ${download_code} EQUAL 0)
                file(REMOVE ${cl2hpp_download_header}) #empty file have to be removed
                message(FATAL_ERROR "Failed to download cl2hpp header")
            endif ()
        endif ()
        get_filename_component(download_dir ${cl2hpp_download_header} DIRECTORY)
        message(STATUS "Found cl2.hpp header at: ${download_dir}")
        if (NOT TARGET OpenCL::cl2hpp)
            add_library(OpenCL::cl2hpp IMPORTED INTERFACE GLOBAL)
            set_target_properties(OpenCL::cl2hpp PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${download_dir}/..")
        endif ()
    endif ()
endif()
