/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <fg/defines.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup surf_functions
 * @{
 */

/**
   Create a Surface object

   \param[out] pSurface will be set to surface handle upon creating the surface object
   \param[in] pXPoints is number of data points along X dimension
   \param[in] pYPoints is number of data points along Y dimension
   \param[in] pType takes one of the values of \ref fg_dtype that indicates
              the integral data type of surface data
   \param[in] pPlotType dictates the type of surface/graph,
              it can take one of the values of \ref fg_plot_type
   \param[in] pMarkerType indicates which symbol is rendered as marker. It can take one of
              the values of \ref fg_marker_type.

   \return \ref fg_err error code
 */
FGAPI fg_err fg_create_surface(fg_surface *pSurface,
                            const unsigned pXPoints, const unsigned pYPoints,
                            const fg_dtype pType,
                            const fg_plot_type pPlotType,
                            const fg_marker_type pMarkerType);

/**
   Destroy surface object

   \param[in] pSurface is the surface handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_destroy_surface(fg_surface pSurface);

/**
   Set the color of surface

   \param[in] pSurface is the surface handle
   \param[in] pRed is Red component in range [0, 1]
   \param[in] pGreen is Green component in range [0, 1]
   \param[in] pBlue is Blue component in range [0, 1]
   \param[in] pAlpha is Blue component in range [0, 1]

   \return \ref fg_err error code
 */
FGAPI fg_err fg_set_surface_color(fg_surface pSurface,
                                  const float pRed, const float pGreen,
                                  const float pBlue, const float pAlpha);

/**
   Set surface legend

   \param[in] pSurface is the surface handle
   \param[in] pLegend

   \return \ref fg_err error code
 */
FGAPI fg_err fg_set_surface_legend(fg_surface pSurface, const char* pLegend);

/**
   Get the resource identifier for vertices buffer

   \param[out] pOut will have the buffer identifier after this function is called
   \param[in] pSurface is the surface handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_surface_vertex_buffer(unsigned* pOut, const fg_surface pSurface);

/**
   Get the resource identifier for colors buffer

   \param[out] pOut will have the buffer identifier after this function is called
   \param[in] pSurface is the surface handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_surface_color_buffer(unsigned* pOut, const fg_surface pSurface);

/**
   Get the resource identifier for alpha values buffer

   \param[out] pOut will have the buffer identifier after this function is called
   \param[in] pSurface is the surface handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_surface_alpha_buffer(unsigned* pOut, const fg_surface pSurface);

/**
   Get the vertices buffer size in bytes

   \param[out] pOut will have the buffer size in bytes after this function is called
   \param[in] pSurface is the surface handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_surface_vertex_buffer_size(unsigned* pOut, const fg_surface pSurface);

/**
   Get the colors buffer size in bytes

   \param[out] pOut will have the buffer size in bytes after this function is called
   \param[in] pSurface is the surface handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_surface_color_buffer_size(unsigned* pOut, const fg_surface pSurface);

/**
   Get the alpha values buffer size in bytes

   \param[out] pOut will have the buffer size in bytes after this function is called
   \param[in] pSurface is the surface handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_surface_alpha_buffer_size(unsigned* pOut, const fg_surface pSurface);

/** @} */

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace forge
{

/**
   \class Surface

   \brief Surface is a graph to display three dimensional data.
 */
class Surface {
    private:
        fg_surface mValue;

    public:
        /**
           Creates a Surface object

           \param[in] pNumXPoints is number of data points along X dimension
           \param[in] pNumYPoints is number of data points along Y dimension
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of surface data
           \param[in] pPlotType is the render type which can be one of \ref PlotType (valid choices
                      are FG_PLOT_SURFACE and FG_PLOT_SCATTER)
           \param[in] pMarkerType is the type of \ref MarkerType to draw for \ref FG_PLOT_SCATTER plot type
         */
        FGAPI Surface(const unsigned pNumXPoints, const unsigned pNumYPoints, const dtype pDataType,
                      const PlotType pPlotType=FG_PLOT_SURFACE, const MarkerType pMarkerType=FG_MARKER_NONE);

        /**
           Copy constructor for surface

           \param[in] pOther is the surface of which we make a copy of.
         */
        FGAPI Surface(const Surface& pOther);

        /**
           surface Destructor
         */
        FGAPI ~Surface();

        /**
           Set the color of line graph(surface)

           \param[in] pColor takes values of forge::Color to define surface color
        */
        FGAPI void setColor(const forge::Color pColor);

        /**
           Set the color of line graph(surface)

           \param[in] pRed is Red component in range [0, 1]
           \param[in] pGreen is Green component in range [0, 1]
           \param[in] pBlue is Blue component in range [0, 1]
           \param[in] pAlpha is Blue component in range [0, 1]
         */
        FGAPI void setColor(const float pRed, const float pGreen,
                            const float pBlue, const float pAlpha);

        /**
           Set surface legend

           \param[in] pLegend
         */
        FGAPI void setLegend(const char* pLegend);

        /**
           Get the buffer identifier for vertices

           \return vertex buffer resource id.
         */
        FGAPI unsigned vertices() const;

        /**
           Get the buffer identifier for color values per vertex

           \return colors buffer resource id.
         */
        FGAPI unsigned colors() const;

        /**
           Get the buffer identifier for alpha values per vertex

           \return alpha values buffer resource id.
         */
        FGAPI unsigned alphas() const;

        /**
           Get the vertex buffer size in bytes

           \return vertex buffer size in bytes
         */
        FGAPI unsigned verticesSize() const;

        /**
           Get the colors buffer size in bytes

           \return colors buffer size in bytes
         */
        FGAPI unsigned colorsSize() const;

        /**
           Get the alpha values buffer size in bytes

           \return alpha buffer size in bytes
         */
        FGAPI unsigned alphasSize() const;


        /**
           Get the handle to internal implementation of surface
         */
        FGAPI fg_surface get() const;
};

}

#endif
