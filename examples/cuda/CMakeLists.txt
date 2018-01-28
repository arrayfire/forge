cmake_minimum_required(VERSION 3.5)

project(Forge-CUDA-Examples LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

if (${CMAKE_VERSION} VERSION_LESS "3.10.0")
    INCLUDE_DIRECTORIES(
          ${Forge_SOURCE_DIR}/include
          ${CUDA_INCLUDE_DIRS}
          ${CMAKE_CURRENT_DIR}
        )
else()
    enable_language(CUDA)
endif ()

macro(make_cuda_example target src)
    add_example(${target} ${src} cuda CXX11
        INCLUDE_DIRS
          ${Forge_SOURCE_DIR}/include
          ${CUDA_INCLUDE_DIRS}
          ${CMAKE_CURRENT_DIR}
        LIBRARIES
          ${CUDA_LIBRARIES}
          ${Forge_LIBRARIES}
    )
endmacro(make_cuda_example)

make_cuda_example(bubblechart bubblechart.cu)
make_cuda_example(field field.cu)
make_cuda_example(fractal fractal.cu)
make_cuda_example(histogram histogram.cu)
make_cuda_example(plot3 plot3.cu)
make_cuda_example(plotting plotting.cu)
make_cuda_example(stream stream.cu)
make_cuda_example(surface surface.cu)
