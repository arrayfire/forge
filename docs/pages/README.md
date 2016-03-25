Overview {#mainpage}
========

[TOC]

## About Forge
A prototype of the OpenGL interop library that can be used with
[ArrayFire](https://github.com/arrayfire/arrayfire). The
goal of Forge is to provide high performance OpenGL visualizations for C/C++
applications that use CUDA/OpenCL.

## Upstream dependencies
* [GLEW](http://glew.sourceforge.net/)
* [GLFW](http://www.glfw.org/)
* [freetype](http://www.freetype.org/)
* On `Linux` and `OS X`, [fontconfig](http://www.freedesktop.org/wiki/Software/fontconfig/) is required.

Above dependecies are available through package managers on most of the
Unix/Linux based distributions. We have provided an option in `CMake` for
`Forge` to build it's own internal `freetype` version if you choose to not
install it on your machine.

We plan to provide support for alternatives to GLFW as windowing toolkit,
however GLFW is the default option. Should you chose to use an alternative, you
have to chose it explicity while building forge.

Currently supported alternatives:
* [SDL2](https://www.libsdl.org/download-2.0.php)

Alternatives to GLFW which are currently under consideration are given below:
* [Qt5](https://wiki.qt.io/Qt_5)

#### Email
* Engineering: technical@arrayfire.com
