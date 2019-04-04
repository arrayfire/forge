cmake_minimum_required(VERSION 3.5)

project(Forge-CUDA-Examples LANGUAGES CXX)

macro(make_cuda_example target src)
  add_example(${target} ${src} cuda CXX11)
endmacro(make_cuda_example)

make_cuda_example(bubblechart bubblechart.cu)
make_cuda_example(field field.cu)
make_cuda_example(fractal fractal.cu)
make_cuda_example(histogram histogram.cu)
make_cuda_example(plot3 plot3.cu)
make_cuda_example(plotting plotting.cu)
make_cuda_example(stream stream.cu)
make_cuda_example(surface surface.cu)
