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

INCLUDE(CMakeParseArguments)

SET(GLSL2CPP_PROGRAM "glsl2cpp")

FUNCTION(GLSL_TO_H)
    CMAKE_PARSE_ARGUMENTS(RTCS "" "VARNAME;EXTENSION;OUTPUT_DIR;TARGETS;NAMESPACE;EOF" "SOURCES" ${ARGN})

    SET(_output_files "")
    FOREACH(_input_file ${RTCS_SOURCES})
        GET_FILENAME_COMPONENT(_path "${_input_file}" PATH)
        GET_FILENAME_COMPONENT(_name "${_input_file}" NAME)
        GET_FILENAME_COMPONENT(var_name "${_input_file}" NAME_WE)

        SET(_namespace "${RTCS_NAMESPACE}")
        STRING(REPLACE "." "_" var_name ${var_name})

        SET(_output_path "${CMAKE_CURRENT_BINARY_DIR}/${RTCS_OUTPUT_DIR}")
        SET(_output_file "${_output_path}/${var_name}.${RTCS_EXTENSION}")

        ADD_CUSTOM_COMMAND(
            OUTPUT ${_output_file}
            DEPENDS ${_input_file} ${GLSL2CPP_PROGRAM}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_output_path}"
            COMMAND ${CMAKE_COMMAND} -E echo "\\#include \\<${_path}/${var_name}.hpp\\>"  >>"${_output_file}"
            COMMAND ${GLSL2CPP_PROGRAM} --file ${_name} --namespace ${_namespace} --output ${_output_file} --name ${var_name} --eof ${RTCS_EOF}
            WORKING_DIRECTORY "${_path}"
            COMMENT "Converting ${_input_file} to GLSL source string"
        )

        LIST(APPEND _output_files ${_output_file})
    ENDFOREACH()
    ADD_CUSTOM_TARGET(${RTCS_NAMESPACE}_bin_target DEPENDS ${_output_files})

    SET("${RTCS_VARNAME}" ${_output_files} PARENT_SCOPE)
    SET("${RTCS_TARGETS}" ${RTCS_NAMESPACE}_bin_target PARENT_SCOPE)
ENDFUNCTION(GLSL_TO_H)
