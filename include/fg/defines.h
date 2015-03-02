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
    FG_SUCCESS=0,
    FG_ERR_INTERNAL,
    FG_ERR_NOMEM,
    FG_ERR_DRIVER,
    FG_ERR_RUNTIME,
    FG_ERR_INVALID_ARRAY,
    FG_ERR_ARG,
    FG_ERR_SIZE,
    FG_ERR_DIFF_TYPE,
    FG_ERR_NOT_SUPPORTED,
    FG_ERR_NOT_CONFIGURED,
    FG_ERR_INVALID_TYPE,
    FG_ERR_INVALID_ARG,
    FG_ERR_GL_ERROR,
    FG_ERR_UNKNOWN
} fg_err;

typedef unsigned int  uint;
typedef unsigned char uchar;

typedef enum {
    FG_RED =1,
    FG_RGB =3,
    FG_RGBA=4,
} fg_color_mode;

// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.

#if defined(_WIN32) || defined(_MSC_VER)
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
    #define STATIC_ static
#else
    #define FGAPI   __attribute__((visibility("default")))
    #include <stdbool.h>
    #define __PRETTY_FUNCTION__ __func__
    #define STATIC_
#endif
