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
class _Plot;
}

namespace fg
{

/**
   \class Plot

   \brief Line graph to display plots.
 */
class Plot {
    private:
        internal::_Plot* value;

    public:
        /**
           Creates a Plot object

           \param[in] pNumPoints is number of data points to display
           \param[in] pDataType takes one of the values of \ref FGType that indicates
                      the integral data type of plot data
         */
        FGAPI Plot(unsigned pNumPoints, FGType pDataType, fg::FGMarkerType=fg::FG_NONE);

        /**
           Copy constructor for Plot

           \param[in] other is the Plot of which we make a copy of.
         */
        FGAPI Plot(const Plot& other);

        /**
           Plot Destructor
         */
        FGAPI ~Plot();

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
        FGAPI void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin);

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
           Get the handle to internal implementation of Histogram
         */
        FGAPI internal::_Plot* get() const;
};

}
