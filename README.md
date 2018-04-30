## Forge - High Performance Visualizations
| Platform | Linux x86 | Linux aarch64 | Windows | OSX |
|:--------:|:---------:|:-------------:|:-------:|:---:|
| Status   | [![Build Status](https://travis-ci.org/arrayfire/forge.svg?branch=master)](https://travis-ci.org/arrayfire/forge) | `Unknown` | [![Build Status](https://ci.appveyor.com/api/projects/status/github/arrayfire/forge?branch=master&svg=true)](https://ci.appveyor.com/project/9prady9/forge-jwb4e) | [![Build Status](https://travis-ci.org/arrayfire/forge.svg?branch=master)](https://travis-ci.org/arrayfire/forge) |


[Join the slack chat channel](https://join.slack.com/t/arrayfire-org/shared_invite/enQtMjI4MjIzMDMzMTczLWM4ODIyZjA3YmY3NWEwMjk2N2Q0YTQyNGMwZmU4ZjkxNGU0MjYzYmUzYTg3ZTM0MDQxOTE2OTJjNGVkOGEwN2M)

An OpenGL interop library that can be used with ArrayFire or any other application using CUDA or OpenCL compute backend. The goal of **Forge** is to provide high performance OpenGL visualizations for C/C++ applications that use CUDA/OpenCL. Forge uses OpenGL >=3.3 forward compatible contexts, so please make sure you have capable hardware before trying it out.

### Documentation

You can find the most recent and updated documentation [here](http://arrayfire.org/forge/index.htm).

### Dependencies
[glbinding](https://github.com/cginternals/glbinding), [GLFW](http://www.glfw.org/), [freetype](http://www.freetype.org/) and [FreeImage](http://freeimage.sourceforge.net/) (optional).

On `Linux` & `OS X`, [fontconfig](http://www.freedesktop.org/wiki/Software/fontconfig/) is required. [SDL2](https://www.libsdl.org/) can be used as an alternative to `glfw`.

Above dependencies are available through package managers on most of the Unix/Linux based distributions. On Windows, we recommend using [vcpkg](https://github.com/Microsoft/vcpkg) to quickly setup forge development environment.

### Sample Images
|     |     |
|-----|-----|
| <img src="./docs/images/image.png" width=150 height=100>Image</img> | <img src="./docs/images/plot.png" width=150 height=100>2D Plot</img>  |
| <img src="./docs/images/plot31.png" width=150 height=100>3d Plot</img> | <img src="./docs/images/plot32.png" width=150 height=100>Rotated 3d Plot</img> |
| <img src="./docs/images/hist.png" width=150 height=100>histogram</img> | <img src="./docs/images/surface.png" width=150 height=100>Surface</img> |
