# Includes the directory if the variable is set
function(conditional_directory variable directory)
  if(${variable})
    add_subdirectory(${directory})
  endif()
endfunction()
