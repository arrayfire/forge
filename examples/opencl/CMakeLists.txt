cmake_minimum_required(VERSION 3.5)

project(Forge-OPENCL-Examples LANGUAGES CXX)

macro(make_ocl_example target src)
    add_example(${target} ${src} opencl CXX11
        INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        LIBRARIES OpenCL::OpenCL;OpenCL::cl2hpp)
endmacro(make_ocl_example)

make_ocl_example(bubblechart bubblechart.cpp)
make_ocl_example(field field.cpp)
make_ocl_example(fractal fractal.cpp)
make_ocl_example(histogram histogram.cpp)
make_ocl_example(plot3 plot3.cpp)
make_ocl_example(plotting plotting.cpp)
make_ocl_example(stream stream.cpp)
make_ocl_example(surface surface.cpp)
