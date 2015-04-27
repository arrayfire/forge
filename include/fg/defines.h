/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER) ||  defined(_WINDOWS_) || defined(__WIN32__) || defined(__WINDOWS__)
#define WINDOWS_OS
#elif defined(__APPLE__) || defined(__MACH__)
#define APPLE_OS
#else
#define LINUX_OS
#endif

#include <GL/glew.h>

#ifdef WINDOWS_OS
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#endif

#ifdef LINUX_OS
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#ifdef WINDOWS_OS
typedef HGLRC ContextHandle;
typedef HDC DisplayHandle;
#endif

#ifdef LINUX_OS
typedef GLXContext ContextHandle;
typedef Display* DisplayHandle;
#endif

// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.

#ifdef WINDOWS_OS
    // http://msdn.microsoft.com/en-us/library/b0084kay(v=VS.80).aspx
    // http://msdn.microsoft.com/en-us/library/3y1sfaz2%28v=VS.80%29.aspx
    #ifdef FGDLL // libfg
        #define FGAPI  __declspec(dllexport)
    #else
        #define FGAPI  __declspec(dllimport)
    #endif

// bool
    #ifndef __cplusplus
        #define bool unsigned char
        #define false 0
        #define true  1
    #endif
    #define __PRETTY_FUNCTION__ __FUNCSIG__
    #define snprintf sprintf_s
    #define FG_STATIC_ static
#else
    #define FGAPI   __attribute__((visibility("default")))
    #include <stdbool.h>
    #define __PRETTY_FUNCTION__ __func__
    #define FG_STATIC_
#endif

namespace fg
{

typedef unsigned int  uint;
typedef unsigned char uchar;

enum ErrorCode {
    FG_SUCCESS            = 0,
    FG_ERR_INTERNAL       = 1,
    FG_ERR_NOMEM          = 2,
    FG_ERR_DRIVER         = 3,
    FG_ERR_RUNTIME        = 4,
    FG_ERR_INVALID_ARRAY  = 5,
    FG_ERR_ARG            = 6,
    FG_ERR_SIZE           = 7,
    FG_ERR_DIFF_TYPE      = 8,
    FG_ERR_NOT_SUPPORTED  = 9,
    FG_ERR_NOT_CONFIGURED = 10,
    FG_ERR_INVALID_TYPE   = 11,
    FG_ERR_INVALID_ARG    = 12,
    FG_ERR_GL_ERROR       = 13,
    FG_ERR_FREETYPE_ERROR = 14,
    FG_ERR_UNKNOWN        = 15
};

enum ColorMode {
    FG_RED =1,
    FG_RGB =3,
    FG_RGBA=4,
};

}
