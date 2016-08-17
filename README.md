Forge - High Performance Visualizations
---------------------------------------

[![Join the chat at https://gitter.im/arrayfire/forge](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/arrayfire/forge?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A prototype of the OpenGL interop library that can be used with ArrayFire. The goal of **Forge** is to provide high performance OpenGL visualizations for C/C++ applications that use CUDA/OpenCL. Forge uses OpenGL >=3.3 forward compatible contexts, so please make sure you have capable hardware before trying it out.

### Build Status
| Platform | Linux x86 | Linux armv7l | Linux aarch64 | Windows | OSX |
|:--------:|:---------:|:------------:|:-------------:|:-------:|:---:|
| Status   | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge/forge-linux)](http://ci.arrayfire.org/view/All/job/forge/job/forge-linux/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge/forge-tegrak1)](http://ci.arrayfire.org/view/All/job/forge/job/forge-tegrak1/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge/forge-tegrax1)](http://ci.arrayfire.org/view/All/job/forge/job/forge-tegrax1/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge/forge-windows)](http://ci.arrayfire.org/view/All/job/forge/job/forge-windows/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=forge/forge-osx)](http://ci.arrayfire.org/view/All/job/forge/job/forge-osx/) |

### Dependencies
* [glbinding](https://github.com/cginternals/glbinding)
* [GLFW](http://www.glfw.org/), optionally you can build with [SDL2](https://www.libsdl.org/) alternative too.
* [freetype](http://www.freetype.org/)
* On `Linux` and `OS X`, [fontconfig](http://www.freedesktop.org/wiki/Software/fontconfig/) is required.

Above dependencies are available through package managers on most of the Unix/Linux based distributions. We have provided an option in `CMake` for `Forge` to build it's own internal `freetype` version if you choose to not install it on your machine.

### Sample Images
|     |     |
|-----|-----|
| <img src="./docs/images/image.png" width=150 height=100>Image</img> | <img src="./docs/images/plot.png" width=150 height=100>2D Plot</img>  |
| <img src="./docs/images/plot31.png" width=150 height=100>3d Plot</img> | <img src="./docs/images/plot32.png" width=150 height=100>Rotated 3d Plot</img> |
| <img src="./docs/images/hist.png" width=150 height=100>histogram</img> | <img src="./docs/images/surface.png" width=150 height=100>Surface</img> |
