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
class _Plot3;
}

namespace fg
{

/**
   \class Plot3

   \brief 3d graph to display 3d line plots.
 */
class Plot3 {
    private:
        internal::_Plot3* value;

    public:
        /**
           Creates a Plot3 object

           \param[in] pNumPoints is number of data points
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of plot data
           \param[in] pPlotType is the render type which can be one of \ref PlotType (valid choices
                      are FG_LINE and FG_SCATTER)
           \param[in] pMarkerType is the type of \ref MarkerType to draw for \ref FG_SCATTER plot type
         */
        FGAPI Plot3(unsigned pNumPoints, dtype pDataType, PlotType pPlotType=fg::FG_LINE, MarkerType pMarkerType=fg::FG_NONE);

        /**
           Copy constructor for Plot3

           \param[in] other is the Plot3 of which we make a copy of.
         */
        FGAPI Plot3(const Plot3& other);

        /**
           Plot3 Destructor
         */
        FGAPI ~Plot3();

        /**
           Set the color of the 3d line plot

           \param[in] col takes values of fg::Color to define plot color
        */
        FGAPI void setColor(fg::Color col);

        /**
           Set the color of the 3d line plot

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
           \param[in] pZmax is Z-Axis maximum value
           \param[in] pZmin is Z-Axis minimum value
         */
        FGAPI void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin, float pZmax, float pZmin);

        /**
           Set axes titles

           \param[in] pXTitle is X-Axis title
           \param[in] pYTitle is Y-Axis title
           \param[in] pZTitle is Z-Axis title
         */
        FGAPI void setAxesTitles(const char* pXTitle, const char* pYTitle, const char* pZTitle);

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
        FGAPI internal::_Plot3* get() const;
};

}
