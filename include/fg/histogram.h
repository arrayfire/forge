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

/** \addtogroup hist_functions
 *  @{
 */

/**
   Creates a Histogram object

   \param[out] pHistogram will point to the histogram object created after this function call
   \param[in] pNBins is number of bins the data is sorted out
   \param[in] pDataType takes one of the values of \ref fg_dtype that indicates
              the integral data type of histogram data

   \return \ref fg_err error code
 */
FGAPI fg_err fg_create_histogram(fg_histogram *pHistogram,
                                 const unsigned pNBins, const fg_dtype pDataType);

/**
   Destroy Histogram object

   \param[in] pHistogram is the histogram handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_destroy_histogram(fg_histogram pHistogram);

/**
   Set the color of bar in the bar graph(histogram)

   This is global alpha value for the histogram rendering that takes
   effect if individual bar alphas are not set by calling the following
   member functions
       - Histogram::alphas()
       - Histogram::alphasSize()

   \param[in] pHistogram is the histogram handle
   \param[in] pRed is Red component in range [0, 1]
   \param[in] pGreen is Green component in range [0, 1]
   \param[in] pBlue is Blue component in range [0, 1]
   \param[in] pAlpha is Alpha component in range [0, 1]

   \return \ref fg_err error code
 */
FGAPI fg_err fg_set_histogram_color(fg_histogram pHistogram,
                                    const float pRed, const float pGreen,
                                    const float pBlue, const float pAlpha);

/**
   Set legend for histogram plot

   \param[in] pHistogram is the histogram handle
   \param[in] pLegend

   \return \ref fg_err error code
 */
FGAPI fg_err fg_set_histogram_legend(fg_histogram pHistogram, const char* pLegend);

/**
   Get the resource identifier for vertices buffer

   \param[out] pOut will have the buffer identifier after this function is called
   \param[in] pHistogram is the histogram handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_histogram_vertex_buffer(unsigned* pOut, const fg_histogram pHistogram);

/**
   Get the resource identifier for colors buffer

   \param[out] pOut will have the buffer identifier after this function is called
   \param[in] pHistogram is the histogram handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_histogram_color_buffer(unsigned* pOut, const fg_histogram pHistogram);

/**
   Get the resource identifier for alpha values buffer

   \param[out] pOut will have the buffer identifier after this function is called
   \param[in] pHistogram is the histogram handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_histogram_alpha_buffer(unsigned* pOut, const fg_histogram pHistogram);

/**
   Get the vertices buffer size in bytes

   \param[out] pOut will have the buffer size in bytes after this function is called
   \param[in] pHistogram is the histogram handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_histogram_vertex_buffer_size(unsigned* pOut, const fg_histogram pHistogram);

/**
   Get the colors buffer size in bytes

   \param[out] pOut will have the buffer size in bytes after this function is called
   \param[in] pHistogram is the histogram handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_histogram_color_buffer_size(unsigned* pOut, const fg_histogram pHistogram);

/**
   Get the alpha values buffer size in bytes

   \param[out] pOut will have the buffer size in bytes after this function is called
   \param[in] pHistogram is the histogram handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_histogram_alpha_buffer_size(unsigned* pOut, const fg_histogram pHistogram);

/** @} */

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

namespace forge
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
           \param[in] pDataType takes one of the values of \ref fg_dtype that indicates
                      the integral data type of histogram data
         */
        FGAPI Histogram(const unsigned pNBins, const dtype pDataType);

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

           \param[in] pColor takes values of type forge::Color to define bar color
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
           Get the handle to internal implementation of Histogram
         */
        FGAPI fg_histogram get() const;
};

}

#endif
