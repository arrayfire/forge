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

/** \addtogroup pie_functions
 *  @{
 */

/**
   Creates a Pie object

   \param[out] pPie will point to the pie object created after this function
   call \param[in] pNSectors is number of sectors the data is sorted out
   \param[in] pDataType takes one of the values of \ref fg_dtype that indicates
              the integral data type of sector data

   \return \ref fg_err error code
 */
FGAPI fg_err fg_create_pie(fg_pie *pPie, const unsigned pNSectors,
                           const fg_dtype pDataType);

/**
   Increase reference count of the resource

   \param[out] pOut is the new handle to existing resource
   \param[in] pIn is the existing resource handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_retain_pie(fg_pie *pOut, fg_pie pIn);

/**
   Destroy Pie object

   \param[in] pPie is the pie handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_release_pie(fg_pie pPie);

/**
   Set the color of bar in the bar graph(pie)

   This is global alpha value for the pie rendering that takes
   effect if individual bar alphas are not set by calling the following
   member functions
       - Pie::alphas()
       - Pie::alphasSize()

   \param[in] pPie is the pie handle
   \param[in] pRed is Red component in range [0, 1]
   \param[in] pGreen is Green component in range [0, 1]
   \param[in] pBlue is Blue component in range [0, 1]
   \param[in] pAlpha is Alpha component in range [0, 1]

   \return \ref fg_err error code
 */
FGAPI fg_err fg_set_pie_color(fg_pie pPie, const float pRed, const float pGreen,
                              const float pBlue, const float pAlpha);

/**
   Set legend for pie plot

   \param[in] pPie is the pie handle
   \param[in] pLegend

   \return \ref fg_err error code
 */
FGAPI fg_err fg_set_pie_legend(fg_pie pPie, const char *pLegend);

/**
   Get the resource identifier for vertices buffer

   \param[out] pOut will have the buffer identifier after this function is
   called \param[in] pPie is the pie handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_pie_vertex_buffer(unsigned *pOut, const fg_pie pPie);

/**
   Get the resource identifier for colors buffer

   \param[out] pOut will have the buffer identifier after this function is
   called \param[in] pPie is the pie handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_pie_color_buffer(unsigned *pOut, const fg_pie pPie);

/**
   Get the resource identifier for alpha values buffer

   \param[out] pOut will have the buffer identifier after this function is
   called \param[in] pPie is the pie handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_pie_alpha_buffer(unsigned *pOut, const fg_pie pPie);

/**
   Get the vertices buffer size in bytes

   \param[out] pOut will have the buffer size in bytes after this function is
   called \param[in] pPie is the pie handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_pie_vertex_buffer_size(unsigned *pOut, const fg_pie pPie);

/**
   Get the colors buffer size in bytes

   \param[out] pOut will have the buffer size in bytes after this function is
   called \param[in] pPie is the pie handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_pie_color_buffer_size(unsigned *pOut, const fg_pie pPie);

/**
   Get the alpha values buffer size in bytes

   \param[out] pOut will have the buffer size in bytes after this function is
   called \param[in] pPie is the pie handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_pie_alpha_buffer_size(unsigned *pOut, const fg_pie pPie);

/** @} */

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace forge {

/// \brief Pie is a statistical graph to display numerical proportion.
class Pie {
private:
  fg_pie mValue;

public:
  /**
     Creates a Pie object

     \param[in] pNSectors is number of bins the data is sorted out
     \param[in] pDataType takes one of the values of \ref fg_dtype that
     indicates the integral data type of pie data
   */
  FGAPI Pie(const unsigned pNSectors, const dtype pDataType);

  /**
     Copy constructor for Pie

     \param[in] pOther is the Pie of which we make a copy of.
   */
  FGAPI Pie(const Pie &pOther);

  /**
    Construct Pie ojbect from fg_pie resource handle

    \param[in] pHandle is the input fg_pie resource handle

    \note This kind of construction assumes ownership of the resource handle
    is released during the Pie object's destruction.
   */
  FGAPI explicit Pie(const fg_pie pHandle);

  /**
     Pie Destructor
   */
  FGAPI ~Pie();

  /**
     Set the color of bar in the bar graph(pie)

     \param[in] pColor takes values of type forge::Color to define bar color
  **/
  FGAPI void setColor(const Color pColor);

  /**
     Set the color of bar in the bar graph(pie)

     This is global alpha value for the pie rendering that takes
     effect if individual bar alphas are not set by calling the following
     member functions
         - Pie::alphas()
         - Pie::alphasSize()

     \param[in] pRed is Red component in range [0, 1]
     \param[in] pGreen is Green component in range [0, 1]
     \param[in] pBlue is Blue component in range [0, 1]
     \param[in] pAlpha is Alpha component in range [0, 1]
   */
  FGAPI void setColor(const float pRed, const float pGreen, const float pBlue,
                      const float pAlpha);

  /**
     Set legend for pie plot

     \param[in] pLegend
   */
  FGAPI void setLegend(const char *pLegend);

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
     Get the handle to internal implementation of Pie
   */
  FGAPI fg_pie get() const;
};

} // namespace forge

#endif
