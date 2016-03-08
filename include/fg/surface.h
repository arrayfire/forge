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

FGAPI fg_err fg_create_surface(fg_surface *pSurface,
                            const uint pXPoints, const uint pYPoints,
                            const fg_dtype pType,
                            const fg_plot_type pPlotType,
                            const fg_marker_type pMarkerType);

FGAPI fg_err fg_destroy_surface(fg_surface pSurface);

FGAPI fg_err fg_set_surface_color(fg_surface pSurface,
                                  const float pRed, const float pGreen,
                                  const float pBlue, const float pAlpha);

FGAPI fg_err fg_set_surface_legend(fg_surface pSurface, const char* pLegend);

FGAPI fg_err fg_get_surface_vbo(uint* pOut, const fg_surface pSurface);

FGAPI fg_err fg_get_surface_cbo(uint* pOut, const fg_surface pSurface);

FGAPI fg_err fg_get_surface_abo(uint* pOut, const fg_surface pSurface);

FGAPI fg_err fg_get_surface_vbo_size(uint* pOut, const fg_surface pSurface);

FGAPI fg_err fg_get_surface_cbo_size(uint* pOut, const fg_surface pSurface);

FGAPI fg_err fg_get_surface_abo_size(uint* pOut, const fg_surface pSurface);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace fg
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
                      the integral data type of plot data
           \param[in] pPlotType is the render type which can be one of \ref PlotType (valid choices
                      are FG_SURFACE and FG_SCATTER)
           \param[in] pMarkerType is the type of \ref MarkerType to draw for \ref FG_SCATTER plot type
         */
        FGAPI Surface(const uint pNumXPoints, const uint pNumYPoints, const dtype pDataType,
                      const PlotType pPlotType=FG_SURFACE, const MarkerType pMarkerType=FG_NONE);

        /**
           Copy constructor for Plot

           \param[in] pOther is the Plot of which we make a copy of.
         */
        FGAPI Surface(const Surface& pOther);

        /**
           Plot Destructor
         */
        FGAPI ~Surface();

        /**
           Set the color of line graph(plot)

           \param[in] pColor takes values of fg::Color to define plot color
        */
        FGAPI void setColor(const fg::Color pColor);

        /**
           Set the color of line graph(plot)

           \param[in] pRed is Red component in range [0, 1]
           \param[in] pGreen is Green component in range [0, 1]
           \param[in] pBlue is Blue component in range [0, 1]
           \param[in] pAlpha is Blue component in range [0, 1]
         */
        FGAPI void setColor(const float pRed, const float pGreen,
                            const float pBlue, const float pAlpha);

        /**
           Set plot legend

           \param[in] pLegend
         */
        FGAPI void setLegend(const char* pLegend);

        /**
           Get the OpenGL buffer object identifier for vertices

           \return OpenGL VBO resource id.
         */
        FGAPI uint vertices() const;

        /**
           Get the OpenGL buffer object identifier for color values per vertex

           \return OpenGL VBO resource id.
         */
        FGAPI uint colors() const;

        /**
           Get the OpenGL buffer object identifier for alpha values per vertex

           \return OpenGL VBO resource id.
         */
        FGAPI uint alphas() const;

        /**
           Get the OpenGL Vertex Buffer Object resource size

           \return vertex buffer object size in bytes
         */
        FGAPI uint verticesSize() const;

        /**
           Get the OpenGL Vertex Buffer Object resource size

           \return colors buffer object size in bytes
         */
        FGAPI uint colorsSize() const;

        /**
           Get the OpenGL Vertex Buffer Object resource size

           \return alpha buffer object size in bytes
         */
        FGAPI uint alphasSize() const;

        /**
           Get the handle to internal implementation of surface
         */
        FGAPI fg_surface get() const;
};

}

#endif
