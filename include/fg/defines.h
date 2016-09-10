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

    #define FG_STATIC_ static
#else
    #define FGAPI   __attribute__((visibility("default")))
    #include <stdbool.h>
    #define FG_STATIC_
#endif

#include <fg/version.h>
#ifndef FG_API_VERSION
#define FG_API_VERSION FG_API_VERSION_CURRENT
#endif

#include <cstdlib>

typedef void* fg_window;
typedef void* fg_font;
typedef void* fg_chart;
typedef void* fg_image;
typedef void* fg_histogram;
typedef void* fg_plot;
typedef void* fg_surface;
typedef void* fg_vector_field;

typedef enum {
    FG_ERR_NONE           = 0,              ///< Fuction returned successfully.
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
     * Font config related error codes
     * '6**'
     * */
    FG_ERR_FONTCONFIG_ERROR = 6001,         ///< Fontconfig related error
    /*
     * FreeImage errors
     */
    FG_ERR_FREEIMAGE_UNKNOWN_FORMAT = 7001, ///< Unknown format, not supported by freeimage
    FG_ERR_FREEIMAGE_BAD_ALLOC = 7002,      ///< freeimage memory allocation failed
    FG_ERR_FREEIMAGE_SAVE_FAILED = 7003,    ///< freeimage file save failed
    /*
     * other error codes
     * match the following pattern
     * '9**'
     * */
    FG_ERR_INTERNAL       = 9001,           ///< Internal error
    FG_ERR_RUNTIME        = 9002,           ///< Runtime error
    FG_ERR_UNKNOWN        = 9003            ///< Unkown error
} fg_err;

typedef enum {
    FG_GRAYSCALE = 100,                     ///< Single channel
    FG_RG        = 200,                     ///< Three(Red, Green & Blue) channels
    FG_RGB       = 300,                     ///< Three(Red, Green & Blue) channels
    FG_BGR       = 301,                     ///< Three(Red, Green & Blue) channels
    FG_RGBA      = 400,                     ///< Four(Red, Green, Blue & Alpha) channels
    FG_BGRA      = 401                      ///< Four(Red, Green, Blue & Alpha) channels
} fg_channel_format;

typedef enum {
    FG_CHART_2D = 2,                        ///< Two dimensional charts
    FG_CHART_3D = 3                         ///< Three dimensional charts
} fg_chart_type;

/**
   Color maps

   \image html gfx_palette.png
 */
typedef enum {
    FG_COLOR_MAP_DEFAULT  = 0,              ///< Default [0-255] grayscale colormap
    FG_COLOR_MAP_SPECTRUM = 1,              ///< Spectrum color
    FG_COLOR_MAP_COLORS   = 2,              ///< Pure Colors
    FG_COLOR_MAP_RED      = 3,              ///< Red color map
    FG_COLOR_MAP_MOOD     = 4,              ///< Mood color map
    FG_COLOR_MAP_HEAT     = 5,              ///< Heat color map
    FG_COLOR_MAP_BLUE     = 6               ///< Blue color map
} fg_color_map;

typedef enum {
    FG_RED     = 0xFF0000FF,
    FG_GREEN   = 0x00FF00FF,
    FG_BLUE    = 0x0000FFFF,
    FG_YELLOW  = 0xFFFF00FF,
    FG_CYAN    = 0x00FFFFFF,
    FG_MAGENTA = 0xFF00FFFF,
    FG_WHITE   = 0xFFFFFFFF,
    FG_BLACK   = 0x000000FF
} fg_color;

typedef enum {
    FG_INT8    = 0,                                ///< Signed byte (8-bits)
    FG_UINT8   = 1,                                ///< Unsigned byte (8-bits)
    FG_INT32   = 2,                                ///< Signed integer (32-bits)
    FG_UINT32  = 3,                                ///< Unsigned integer (32-bits)
    FG_FLOAT32 = 4,                                ///< Float (32-bits)
    FG_INT16   = 5,                                ///< Signed integer (16-bits)
    FG_UINT16  = 6                                 ///< Unsigned integer (16-bits)
} fg_dtype;

typedef enum {
    FG_PLOT_LINE         = 0,                    ///< Line plot
    FG_PLOT_SCATTER      = 1,                    ///< Scatter plot
    FG_PLOT_SURFACE      = 2                     ///< Surface plot
} fg_plot_type;

typedef enum {
    FG_MARKER_NONE         = 0,                    ///< No marker
    FG_MARKER_POINT        = 1,                    ///< Point marker
    FG_MARKER_CIRCLE       = 2,                    ///< Circle marker
    FG_MARKER_SQUARE       = 3,                    ///< Square marker
    FG_MARKER_TRIANGLE     = 4,                    ///< Triangle marker
    FG_MARKER_CROSS        = 5,                    ///< Cross-hair marker
    FG_MARKER_PLUS         = 6,                    ///< Plus symbol marker
    FG_MARKER_STAR         = 7                     ///< Star symbol marker
} fg_marker_type;


#ifdef __cplusplus
namespace forge
{
    typedef fg_err ErrorCode;
    typedef fg_channel_format ChannelFormat;
    typedef fg_chart_type ChartType;
    typedef fg_color_map ColorMap;
    typedef fg_color Color;
    typedef fg_plot_type PlotType;
    typedef fg_marker_type MarkerType;

    typedef enum {
        s8  = FG_INT8,
        u8  = FG_UINT8,
        s32 = FG_INT32,
        u32 = FG_UINT32,
        f32 = FG_FLOAT32,
        s16 = FG_INT16,
        u16 = FG_UINT16,
    } dtype;
}
#endif
