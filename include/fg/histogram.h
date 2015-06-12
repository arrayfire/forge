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
class _Histogram;
}

namespace fg
{

/**
   \class Histogram

   \brief Bar graph to display data frequencey.
 */
class Histogram {
    private:
        internal::_Histogram* value;

    public:
        /**
           Creates a Histogram object

           \param[in] pNBins is number of bins the data is sorted out
           \param[in] pDataType takes one of the values of \ref FGType that indicates
                      the integral data type of histogram data
         */
        FGAPI Histogram(unsigned pNBins, FGType pDataType);

        /**
           Copy constructor for Histogram

           \param[in] other is the Histogram of which we make a copy of.
         */
        FGAPI Histogram(const Histogram& other);

        /**
           Histogram Destructor
         */
        FGAPI ~Histogram();

        /**
           Set the color of bar in the bar graph(histogram)

           \param[in] pRed is Red component in range [0, 1]
           \param[in] pGreen is Green component in range [0, 1]
           \param[in] pBlue is Blue component in range [0, 1]
         */
        FGAPI void setBarColor(float pRed, float pGreen, float pBlue);

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
        FGAPI internal::_Histogram* get() const;
};

}
