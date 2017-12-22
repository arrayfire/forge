cmake_minimum_required(VERSION 3.5)

project(Forge-CPU-Examples LANGUAGES CXX)

add_example(bubblechart bubblechart.cpp cpu CXX11)
add_example(field field.cpp cpu)
add_example(fractal fractal.cpp cpu)
add_example(histogram histogram.cpp cpu)
add_example(plot3 plot3.cpp cpu)
add_example(plotting plotting.cpp cpu)
add_example(stream stream.cpp cpu)
add_example(surface surface.cpp cpu)
