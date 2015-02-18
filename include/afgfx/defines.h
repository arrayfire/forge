/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#ifdef __cplusplus
#endif

typedef enum {
    AFGFX_SUCCESS=0,
    AFGFX_ERR_INTERNAL,
    AFGFX_ERR_NOMEM,
    AFGFX_ERR_DRIVER,
    AFGFX_ERR_RUNTIME,
    AFGFX_ERR_INVALID_ARRAY,
    AFGFX_ERR_ARG,
    AFGFX_ERR_SIZE,
    AFGFX_ERR_DIFF_TYPE,
    AFGFX_ERR_NOT_SUPPORTED,
    AFGFX_ERR_NOT_CONFIGURED,
    AFGFX_ERR_INVALID_TYPE,
    AFGFX_ERR_INVALID_ARG,
    AFGFX_ERR_GL_ERROR,
    AFGFX_ERR_UNKNOWN
} afgfx_err;

typedef unsigned int  uint;
typedef unsigned char uchar;

typedef enum {
    AFGFX_RED =1,
    AFGFX_RGB =3,
    AFGFX_RGBA=4,
} afgfx_color_mode;

// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.

#if defined(_WIN32) || defined(_MSC_VER)
    // http://msdn.microsoft.com/en-us/library/b0084kay(v=VS.80).aspx
    // http://msdn.microsoft.com/en-us/library/3y1sfaz2%28v=VS.80%29.aspx
    #ifdef AFGFXDLL // libafgfx
        #define AFGFXAPI  __declspec(dllexport)
    #else
        #define AFGFXAPI  __declspec(dllimport)
    #endif

// bool
    #ifndef __cplusplus
        #define bool unsigned char
        #define false 0
        #define true  1
    #endif
    #define __PRETTY_FUNCTION__ __FUNCSIG__
    #define snprintf sprintf_s
    #define STATIC_ static
#else
    #define AFGFXAPI   __attribute__((visibility("default")))
    #include <stdbool.h>
    #define __PRETTY_FUNCTION__ __func__
    #define STATIC_
#endif
