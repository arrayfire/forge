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

FGAPI fg_err fg_create_histogram(fg_histogram *pHistogram,
                                 const uint nbins, const fg_dtype type);

FGAPI fg_err fg_destroy_histogram(fg_histogram pHistogram);

FGAPI fg_err fg_set_histogram_color(fg_histogram pHistogram,
                                    const float pRed, const float pGreen,
                                    const float pBlue, const float pAlpha);

FGAPI fg_err fg_set_histogram_legend(fg_histogram pHistogram, const char* pLegend);

FGAPI fg_err fg_get_histogram_vbo(uint* out, fg_histogram pHistogram);

FGAPI fg_err fg_get_histogram_cbo(uint* out, fg_histogram pHistogram);

FGAPI fg_err fg_get_histogram_abo(uint* out, fg_histogram pHistogram);

FGAPI fg_err fg_get_histogram_vbo_size(uint* out, fg_histogram pHistogram);

FGAPI fg_err fg_get_histogram_cbo_size(uint* out, fg_histogram pHistogram);

FGAPI fg_err fg_get_histogram_abo_size(uint* out, fg_histogram pHistogram);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

namespace fg
{

/**
   \class Histogram

   \brief Histogram is a bar graph to display data frequencey.
 */
class Histogram {
    private:
        fg_histogram mValue;

    public:
        /**
           Creates a Histogram object

           \param[in] pNBins is number of bins the data is sorted out
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of histogram data
         */
        FGAPI Histogram(const uint pNBins, const dtype pDataType);

        /**
           Copy constructor for Histogram

           \param[in] pOther is the Histogram of which we make a copy of.
         */
        FGAPI Histogram(const Histogram& pOther);

        /**
           Histogram Destructor
         */
        FGAPI ~Histogram();

        /**
           Set the color of bar in the bar graph(histogram)

           \param[in] pColor takes values of type fg::Color to define bar color
        **/
        FGAPI void setColor(const Color pColor);

        /**
           Set the color of bar in the bar graph(histogram)

           This is global alpha value for the histogram rendering that takes
           effect if individual bar alphas are not set by calling the following
           member functions
               - Histogram::alphas()
               - Histogram::alphasSize()

           \param[in] pRed is Red component in range [0, 1]
           \param[in] pGreen is Green component in range [0, 1]
           \param[in] pBlue is Blue component in range [0, 1]
           \param[in] pAlpha is Alpha component in range [0, 1]
         */
        FGAPI void setColor(const float pRed, const float pGreen,
                            const float pBlue, const float pAlpha);

        /**
           Set legend for histogram plot

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
           Get the handle to internal implementation of Histogram
         */
        FGAPI fg_histogram get() const;
};

}

#endif
