# Includes the cmake script if the variable is NOT true
macro(include_if_not variable cmake_script)
    if(NOT ${variable})
    include(${cmake_script})
  endif()
endmacro()

