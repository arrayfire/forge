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
    FW_SUCCESS=0,
    FW_ERR_INTERNAL,
    FW_ERR_NOMEM,
    FW_ERR_DRIVER,
    FW_ERR_RUNTIME,
    FW_ERR_INVALID_ARRAY,
    FW_ERR_ARG,
    FW_ERR_SIZE,
    FW_ERR_DIFF_TYPE,
    FW_ERR_NOT_SUPPORTED,
    FW_ERR_NOT_CONFIGURED,
    FW_ERR_INVALID_TYPE,
    FW_ERR_INVALID_ARG,
    FW_ERR_GL_ERROR,
    FW_ERR_UNKNOWN
} fw_err;

typedef enum {
    f32,
    c32,
    f64,
    c64,
    b8,
    s32,
    u32,
    u8,
    u8x4,
} fw_dtype;

typedef enum {
    FW_GREY=1,
    FW_RGB =3,
    FW_RGBA=4,
} fw_color_mode;

typedef struct
{
    GLFWwindow*     pWindow;
    GLEWContext*    pGLEWContext;
    int             uiWidth;
    int             uiHeight;
    int             uiID;
} fw_window;

typedef fw_window* WindowHandle;

typedef struct
{
    WindowHandle *window;

    //OpenGL PBO and texture "names"
    GLuint gl_PBO;
    GLuint gl_Tex;
    GLuint gl_Shader;
} fw_image;

// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.

#if defined(_WIN32) || defined(_MSC_VER)
    // http://msdn.microsoft.com/en-us/library/b0084kay(v=VS.80).aspx
    // http://msdn.microsoft.com/en-us/library/3y1sfaz2%28v=VS.80%29.aspx
    #ifdef FWDLL // libfw
        #define FWAPI  __declspec(dllexport)
    #else
        #define FWAPI  __declspec(dllimport)
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
    #define FWAPI   __attribute__((visibility("default")))
    #include <stdbool.h>
    #define __PRETTY_FUNCTION__ __func__
    #define STATIC_
#endif
