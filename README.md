Forge - High Performance Visualizations
---------------------------------------

[![Join the chat at https://gitter.im/arrayfire/forge](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/arrayfire/forge?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A prototype of the OpenGL interop library that can be used with ArrayFire. The goal of `Forge` is to provide high performance OpenGL visualizations for C/C++ applications that use CUDA/OpenCL. Forge uses OpenGL >=3.3 forward compatible contexts, so please make sure you have capable hardware before trying it out.

### Build Status
| Platform | Linux x86 | Linux armv7l | Linux aarch64 | Windows | OSX |
|:--------:|:---------:|:------------:|:-------------:|:-------:|:---:|
| Status   | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge-linux)](http://ci.arrayfire.org/job/forge-linux) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge-tegrak1)](http://ci.arrayfire.org/job/forge-tegrak1) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge-tegrax1)](http://ci.arrayfire.org/job/forge-tegrax1) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge-windows)](http://ci.arrayfire.org/job/forge-windows) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge-osx)](http://ci.arrayfire.org/job/forge-osx) |

### Dependencies
* [GLEW](http://glew.sourceforge.net/)
* [GLFW](http://www.glfw.org/)
* [freetype](http://www.freetype.org/)
* On `Linux` and `OS X`, [fontconfig](http://www.freedesktop.org/wiki/Software/fontconfig/) is required.

Above dependecies are available through package managers on most of the Unix/Linux based distributions. We have provided an option in `CMake` for `Forge` to build it's own internal `freetype` version if you choose to not install it on your machine.
