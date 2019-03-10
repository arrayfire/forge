cmake_minimum_required(VERSION 3.5)

project(Forge-CUDA-Examples LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

macro(make_cuda_example target src)
    add_example(${target} ${src} cuda CXX11
        INCLUDE_DIRS
          ${CUDA_INCLUDE_DIRS}
          ${CMAKE_CURRENT_SOURCE_DIR}
        LIBRARIES
          ${CUDA_LIBRARIES})
endmacro(make_cuda_example)

make_cuda_example(bubblechart bubblechart.cu)
make_cuda_example(field field.cu)
make_cuda_example(fractal fractal.cu)
make_cuda_example(histogram histogram.cu)
make_cuda_example(plot3 plot3.cu)
make_cuda_example(plotting plotting.cu)
make_cuda_example(stream stream.cu)
make_cuda_example(surface surface.cu)
