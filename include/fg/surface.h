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

#include <string>

namespace internal
{
class _Surface;
}

namespace fg
{

class Window;

/**
   \class Surface

   \brief 3d graph to display plots.
 */
class Surface {
    private:
        internal::_Surface* mValue;

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

           \param[in] other is the Plot of which we make a copy of.
         */
        FGAPI Surface(const Surface& pOther);

        /**
           Plot Destructor
         */
        FGAPI ~Surface();

        /**
           Set the color of line graph(plot)

           \param[in] col takes values of fg::Color to define plot color
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
        FGAPI void setLegend(const std::string& pLegend);

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
        FGAPI internal::_Surface* get() const;
};

}
