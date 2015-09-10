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

/**
  Requirment by GLEWmx
 */
FGAPI GLEWContext* glewGetContext();

namespace fg
{

enum ErrorCode {
    FG_SUCCESS            = 0,              ///< Fuction returned successfully.
    /*
     * Arguement related error codes that are
     * generated when invalid arguments are
     * provided to any function. All these
     * codes match the following pattern
     * '1***'
     * */
    FG_ERR_SIZE           = 1001,           ///< Invalid size argument
    FG_ERR_INVALID_TYPE   = 1002,           ///< Invalid type argument
    FG_ERR_INVALID_ARG    = 1003,           ///< Invalid argument
    /*
     * OpenGL related error codes
     * match the following pattern
     * '2***'
     * */
    FG_ERR_GL_ERROR       = 2001,           ///< OpenGL error
    /*
     * FreeType related error codes
     * match the following pattern
     * '3***'
     * */
    FG_ERR_FREETYPE_ERROR = 3001,           ///< Freetype library error
    /*
     * File IO related error codes
     * match the following pattern
     * '4***'
     * */
    FG_ERR_FILE_NOT_FOUND = 4001,           ///< File IO errors
    /*
     * Unsupported configurations
     * and other similar error codes
     * match the following pattern
     * '5***'
     * */
    FG_ERR_NOT_SUPPORTED  = 5001,           ///< Feature not supported
    FG_ERR_NOT_CONFIGURED = 5002,           ///< Library configuration mismatch
    /*
     * other error codes
     * match the following pattern
     * '9**'
     * */
    FG_ERR_INTERNAL       = 9001,           ///< Internal error
    FG_ERR_RUNTIME        = 9002,           ///< Runtime error
    FG_ERR_UNKNOWN        = 9003            ///< Unkown error
};

enum ColorMode {
    FG_RED =1,                              ///< Single channel
    FG_RGB =3,                              ///< Three(Red, Green & Blue) channels
    FG_RGBA=4                               ///< Four(Red, Green, Blue & Alpha) channels
};

/**
   Color maps

   \image html gfx_palette.png
 */
enum ColorMap {
    FG_DEFAULT  = 0,                        ///< Default [0-255] grayscale colormap
    FG_SPECTRUM = 1,                        ///< Spectrum color
    FG_COLORS   = 2,                        ///< Pure Colors
    FG_REDMAP   = 3,                        ///< Red color map
    FG_MOOD     = 4,                        ///< Mood color map
    FG_HEAT     = 5,                        ///< Heat color map
    FG_BLUEMAP  = 6                         ///< Blue color map
};

enum FGType {
    FG_BYTE         = 0,                    ///< Signed byte (8-bits)
    FG_UNSIGNED_BYTE= 1,                    ///< Unsigned byte (8-bits)
    FG_INT          = 2,                    ///< Signed integer (32-bits)
    FG_UNSIGNED_INT = 3,                    ///< Unsigned integer (32-bits)
    FG_FLOAT        = 4                     ///< Float (32-bits)
};

enum FGMarkerType {
    FG_NONE         = 0,
    FG_POINT        = 1,
    FG_CIRCLE       = 2,
    FG_TRIANGLE     = 3,
    FG_STAR         = 4,
    FG_SQUARE       = 5
};

}
