# Function to turn an GLSL shader source file into a C string within a source file.
# xxd uses its input's filename to name the string and its length, so we
# need to move them to a name that depends only on the path output, not its
# input.  Otherwise, builds in different relative locations would put the
# source into different variable names, and everything would fall over.
# The actual name will be filename (.s replaced with underscores), and length
# name_len.
#
# Usage example:
#
# set(KERNELS a.cl b/c.cl)
# resource_to_cxx_source(
#   SOURCES ${KERNELS}
#   VARNAME OUTPUTS
# )
# add_executable(foo ${OUTPUTS})
#
# The namespace they are placed in is taken from filename.namespace.
#
# For example, if the input file is kernel.cl, the two variables will be
#  unsigned char ns::kernel_cl[];
#  unsigned int ns::kernel_cl_len;
#
# where ns is the contents of kernel.cl.namespace.

set(GLSL2CPP_PROGRAM "glsl2cpp")

macro(convert_glsl_shaders_to_headers)
    cmake_parse_arguments(
        RTCS "" "VARNAME;EXTENSION;OUTPUT_DIR;TARGETS;NAMESPACE;EOF" "SOURCES" ${ARGN})

    set(_output_files "")
    foreach(_input_file ${RTCS_SOURCES})
        get_filename_component(_path "${_input_file}" PATH)
        get_filename_component(_name "${_input_file}" NAME)
        get_filename_component(var_name "${_input_file}" NAME_WE)

        set(_namespace "${RTCS_NAMESPACE}")
        string(REPLACE "." "_" var_name ${var_name})

        set(_output_path "${PROJECT_BINARY_DIR}/${RTCS_OUTPUT_DIR}")
        set(_output_file "${_output_path}/${var_name}.${RTCS_EXTENSION}")

        add_custom_command(
            OUTPUT ${_output_file}
            DEPENDS ${_input_file} ${GLSL2CPP_PROGRAM}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_output_path}"
            COMMAND ${CMAKE_COMMAND} -E echo "\\#include \\<${_path}/${var_name}.hpp\\>"  >>"${_output_file}"
            COMMAND ${GLSL2CPP_PROGRAM} --file ${_name} --namespace ${_namespace} --output ${_output_file} --name ${var_name} --eof ${RTCS_EOF}
            WORKING_DIRECTORY "${_path}"
            COMMENT "Converting ${_input_file} to GLSL source string"
        )

        list(APPEND _output_files ${_output_file})
    endforeach()
    add_custom_target(${RTCS_NAMESPACE}_bin_target DEPENDS ${_output_files})

    set("${RTCS_VARNAME}" ${_output_files} PARENT_SCOPE)
    set("${RTCS_TARGETS}" ${RTCS_NAMESPACE}_bin_target PARENT_SCOPE)
endmacro(convert_glsl_shaders_to_headers)
