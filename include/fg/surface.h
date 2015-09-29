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

namespace internal
{
class _Surface;
}

namespace fg
{

/**
   \class Surface

   \brief 3d graph to display plots.
 */
class Surface {
    private:
        internal::_Surface* value;

    public:
        /**
           Creates a Surface object

           \param[in] pNumXPoints is number of data points along X dimension
           \param[in] pNumYPoints is number of data points along Y dimension
           \param[in] pDataType takes one of the values of \ref FGType that indicates
                      the integral data type of plot data
         */
        FGAPI Surface(unsigned pNumXPoints, unsigned pNumYPoints, dtype pDataType, PlotType pPlotType=fg::FG_SURFACE, MarkerType pMarkerType=fg::FG_NONE);

        /**
           Copy constructor for Plot

           \param[in] other is the Plot of which we make a copy of.
         */
        FGAPI Surface(const Surface& other);

        /**
           Plot Destructor
         */
        FGAPI ~Surface();

        /**
           Set the color of line graph(plot)

           \param[in] col takes values of fg::Color to define plot color
        */
        FGAPI void setColor(fg::Color col);

        /**
           Set the color of line graph(plot)

           \param[in] pRed is Red component in range [0, 1]
           \param[in] pGreen is Green component in range [0, 1]
           \param[in] pBlue is Blue component in range [0, 1]
         */
        FGAPI void setColor(float pRed, float pGreen, float pBlue);

        /**
           Set the chart axes limits

           \param[in] pXmax is X-Axis maximum value
           \param[in] pXmin is X-Axis minimum value
           \param[in] pYmax is Y-Axis maximum value
           \param[in] pYmin is Y-Axis minimum value
         */
        FGAPI void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin, float pZmax, float pZmin);

        /**
           Set X-Axis title in chart

           \param[in] pTitle is axis title
         */
        FGAPI void setXAxisTitle(const char* pTitle);

        /**
           Set Y-Axis title in chart

           \param[in] pTitle is axis title
         */
        FGAPI void setYAxisTitle(const char* pTitle);

        /**
           Set Z-Axis title in chart

           \param[in] pTitle is axis title
         */
        FGAPI void setZAxisTitle(const char* pTitle);

        /**
           Get X-Axis maximum value

           \return Maximum value along X-Axis
         */
        FGAPI float xmax() const;

        /**
           Get X-Axis minimum value

           \return Minimum value along X-Axis
         */
        FGAPI float xmin() const;

        /**
           Get Y-Axis maximum value

           \return Maximum value along Y-Axis
         */
        FGAPI float ymax() const;

        /**
           Get Y-Axis minimum value

           \return Minimum value along Y-Axis
         */
        FGAPI float ymin() const;

        /**
           Get Z-Axis maximum value

           \return Maximum value along Z-Axis
         */
        FGAPI float zmax() const;

        /**
           Get Z-Axis minimum value

           \return Minimum value along Z-Axis
         */
        FGAPI float zmin() const;

        /**
           Get the OpenGL Vertex Buffer Object identifier

           \return OpenGL VBO resource id.
         */
        FGAPI unsigned vbo() const;

        /**
           Get the OpenGL Vertex Buffer Object resource size

           \return OpenGL VBO resource size.
         */
        FGAPI unsigned size() const;

        /**
           Get the handle to internal implementation of _Surface
         */
        FGAPI internal::_Surface* get() const;
};

}
