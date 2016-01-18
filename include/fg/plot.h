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
class _Plot;
}

namespace fg
{

class Window;

/**
   \class Plot

   \brief Line graph to display plots.
 */
class Plot {
    private:
        internal::_Plot* mValue;

    public:
        /**
           Creates a Plot object

           \param[in] pNumPoints is number of data points to display
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of plot data
         */
        FGAPI Plot(const uint pNumPoints, const dtype pDataType, const ChartType pChartType,
                   const PlotType=FG_LINE, const MarkerType=FG_NONE);

        /**
           Copy constructor for Plot

           \param[in] other is the Plot of which we make a copy of.
         */
        FGAPI Plot(const Plot& pOther);

        /**
           Plot Destructor
         */
        FGAPI ~Plot();

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
           Get the handle to internal implementation of plot
         */
        FGAPI internal::_Plot* get() const;
};

}
