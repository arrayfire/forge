/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef OS_WIN
    // http://msdn.microsoft.com/en-us/library/b0084kay(v=VS.80).aspx
    // http://msdn.microsoft.com/en-us/library/3y1sfaz2%28v=VS.80%29.aspx
    #ifdef FGDLL // libfg
        #define FGAPI  __declspec(dllexport)
    #else
        #define FGAPI  __declspec(dllimport)
    #endif

    #ifndef __cplusplus
        #define false 0
        #define true  1
    #endif

    #define __PRETTY_FUNCTION__ __FUNCSIG__
    #if _MSC_VER < 1900
        #define snprintf sprintf_s
    #endif
    #define FG_STATIC_ static
#else
    #define FGAPI   __attribute__((visibility("default")))
    #include <stdbool.h>
    #define __PRETTY_FUNCTION__ __func__
    #define FG_STATIC_
#endif

#include <GL/glew.h>

// Required to be defined for GLEW MX to work,
// // along with the GLEW_MX define in the perprocessor!
FGAPI GLEWContext* glewGetContext();

namespace fg
{

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

enum ColorMap {
    FG_DEFAULT  = 0,
    FG_SPECTRUM = 1,
    FG_COLORS   = 2,
    FG_REDMAP   = 3,
    FG_MOOD     = 4,
    FG_HEAT     = 5,
    FG_BLUEMAP  = 6
};

enum FGType {
    FG_BYTE         = 0,
    FG_UNSIGNED_BYTE= 1,
    FG_INT          = 2,
    FG_UNSIGNED_INT = 3,
    FG_FLOAT        = 4,
};

}
