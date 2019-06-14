/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined(OS_WIN)
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
#else
    #define FGAPI   __attribute__((visibility("default")))
    #include <stdbool.h>
#endif

#include <fg/version.h>
#ifndef FG_API_VERSION
#define FG_API_VERSION FG_API_VERSION_CURRENT
#endif

#include <cstdlib>

/// \brief Window handle
typedef void* fg_window;

/// \brief Font handle
typedef void* fg_font;

/// \brief Chart handle
typedef void* fg_chart;

/// \brief Image handle
typedef void* fg_image;

/// \brief Histogram handle
typedef void* fg_histogram;

/// \brief Plot handle
typedef void* fg_plot;

/// \brief Surface handle
typedef void* fg_surface;

/// \brief Vector Field handle
typedef void* fg_vector_field;

/// \brief Return Error Codes for Forge C API
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

/// \brief Image Channel Formats
typedef enum {
    FG_GRAYSCALE = 100,                     ///< Single channel
    FG_RG        = 200,                     ///< Three(Red, Green & Blue) channels
    FG_RGB       = 300,                     ///< Three(Red, Green & Blue) channels
    FG_BGR       = 301,                     ///< Three(Red, Green & Blue) channels
    FG_RGBA      = 400,                     ///< Four(Red, Green, Blue & Alpha) channels
    FG_BGRA      = 401                      ///< Four(Red, Green, Blue & Alpha) channels
} fg_channel_format;

/// \brief Chart dimensionality i.e. 2D or 3D
typedef enum {
    FG_CHART_2D = 2,                        ///< Two dimensional charts
    FG_CHART_3D = 3                         ///< Three dimensional charts
} fg_chart_type;

/// \brief Color Maps
typedef enum {
    FG_COLOR_MAP_DEFAULT  =  0,              ///< Default [0-255] grayscale colormap
    FG_COLOR_MAP_SPECTRUM =  1,              ///< Visual spectrum (390nm-830nm) in sRGB colorspace
    FG_COLOR_MAP_RAINBOW  =  2,              ///< Rainbow color map
    FG_COLOR_MAP_RED      =  3,              ///< Red color map
    FG_COLOR_MAP_MOOD     =  4,              ///< Mood color map
    FG_COLOR_MAP_HEAT     =  5,              ///< Heat color map
    FG_COLOR_MAP_BLUE     =  6,              ///< Blue color map
    FG_COLOR_MAP_INFERNO  =  7,              ///< perceptually uniform shades of black-red-yellow
    FG_COLOR_MAP_MAGMA    =  8,              ///< perceptually uniform shades of black-red-white
    FG_COLOR_MAP_PLASMA   =  9,              ///< perceptually uniform shades of blue-red-yellow
    FG_COLOR_MAP_VIRIDIS  = 10,              ///< perceptually uniform shades of blue-green-yellow
} fg_color_map;

/// \brief Color Constants
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

/// \brief Enum representation of internal data types
typedef enum {
    FG_INT8    = 0,                                ///< Signed byte (8-bits)
    FG_UINT8   = 1,                                ///< Unsigned byte (8-bits)
    FG_INT32   = 2,                                ///< Signed integer (32-bits)
    FG_UINT32  = 3,                                ///< Unsigned integer (32-bits)
    FG_FLOAT32 = 4,                                ///< Float (32-bits)
    FG_INT16   = 5,                                ///< Signed integer (16-bits)
    FG_UINT16  = 6                                 ///< Unsigned integer (16-bits)
} fg_dtype;

/// \brief Plot Style
typedef enum {
    FG_PLOT_LINE         = 0,                    ///< Line plot
    FG_PLOT_SCATTER      = 1,                    ///< Scatter plot
    FG_PLOT_SURFACE      = 2                     ///< Surface plot
} fg_plot_type;

/// \brief Markers rendered as sprites
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

/// \brief Forge API namespace
namespace forge {
    typedef fg_err ErrorCode;
    typedef fg_channel_format ChannelFormat;
    typedef fg_chart_type ChartType;
    typedef fg_color_map ColorMap;
    typedef fg_color Color;
    typedef fg_plot_type PlotType;
    typedef fg_marker_type MarkerType;

    /// \brief Alias Enum to \ref fg_dtype enum
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
