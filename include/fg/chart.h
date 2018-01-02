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
#include <fg/image.h>
#include <fg/plot.h>
#include <fg/surface.h>
#include <fg/vector_field.h>
#include <fg/histogram.h>


#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup chart_functions
 *  @{
 */

/**
   Create a Chart object with given dimensional property

   \param[out] pHandle will be set to point to the chart object in memory
   \param[in] pChartType is chart dimension property

   \return \ref fg_err error code
 */
FGAPI fg_err fg_create_chart(fg_chart *pHandle,
                             const fg_chart_type pChartType);

/**
   Increase reference count for the chart resource

   \param[out] pOut is new reference to existing resource
   \param[in] pChart is the existing chart resource handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_retain_chart(fg_chart* pOut, fg_chart pChart);

/**
   Destroy the chart object

   \param[in] pHandle is chart handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_release_chart(fg_chart pHandle);

/**
   Set axes titles for the chart

   \param[in] pHandle is chart handle
   \param[in] pX is x-axis title label
   \param[in] pY is y-axis title label
   \param[in] pZ is z-axis title label

   \return \ref fg_err error code
 */
FGAPI fg_err fg_set_chart_axes_titles(fg_chart pHandle,
                                      const char* pX,
                                      const char* pY,
                                      const char* pZ);

/**
   Set axes data ranges

   \param[in] pHandle is chart handle
   \param[in] pXmin is x-axis minimum data value
   \param[in] pXmax is x-axis maximum data value
   \param[in] pYmin is y-axis minimum data value
   \param[in] pYmax is y-axis maximum data value
   \param[in] pZmin is z-axis minimum data value
   \param[in] pZmax is z-axis maximum data value
 */
FGAPI fg_err fg_set_chart_axes_limits(fg_chart pHandle,
                                      const float pXmin, const float pXmax,
                                      const float pYmin, const float pYmax,
                                      const float pZmin, const float pZmax);

/**
   Set the format for display of axes labels in a chart

   \param[in] pHandle is chart handle
   \param[in] pXFormat sets the display format for numbers of X axis
   \param[in] pYFormat sets the display format for numbers of Y axis
   \param[in] pZFormat sets the display format for numbers of Z axis

   Display format string follows printf style formating for numbers
 */
FGAPI fg_err fg_set_chart_label_format(fg_chart pHandle, const char* pXFormat,
                                       const char* pYFormat, const char* pZFormat);

/**
   Get axes data ranges

   \param[out] pXmin is x-axis minimum data value
   \param[out] pXmax is x-axis maximum data value
   \param[out] pYmin is y-axis minimum data value
   \param[out] pYmax is y-axis maximum data value
   \param[out] pZmin is z-axis minimum data value
   \param[out] pZmax is z-axis maximum data value
   \param[in] pHandle is chart handle
 */
FGAPI fg_err fg_get_chart_axes_limits(float* pXmin, float* pXmax,
                                      float* pYmin, float* pYmax,
                                      float* pZmin, float* pZmax,
                                      const fg_chart pHandle);
/**
   Set legend position for Chart

   \param[in] pHandle is chart handle
   \param[in] pX is horizontal position in normalized coordinates
   \param[in] pY is vertical position in normalized coordinates

   \return \ref fg_err error code

   \note By normalized coordinates, the range of these coordinates is expected to be [0-1].
   (0,0) is the bottom hand left corner.
 */
FGAPI fg_err fg_set_chart_legend_position(fg_chart pHandle, const float pX, const float pY);

/**
   Add an existing image object to chart

   \param[in] pChart is the chart to which image has to be added
   \param[in] pImage is the image to be added to the chart

   \return \ref fg_err error code
 */
FGAPI fg_err fg_append_image_to_chart(fg_chart pChart, fg_image pImage);

/**
   Add an existing histogram object to chart

   \param[in] pChart is the chart to which histogram has to be added
   \param[in] pHistogram is the histogram to be added to the chart

   \return \ref fg_err error code
 */
FGAPI fg_err fg_append_histogram_to_chart(fg_chart pChart, fg_histogram pHistogram);

/**
   Add an existing plot object to chart

   \param[in] pChart is the chart to which plot has to be added
   \param[in] pPlot is the plot to be added to the chart

   \return \ref fg_err error code
 */
FGAPI fg_err fg_append_plot_to_chart(fg_chart pChart, fg_plot pPlot);

/**
   Add an existing surface object to chart

   \param[in] pChart is the chart to which surface has to be added
   \param[in] pSurface is the surface to be added to the chart

   \return \ref fg_err error code
 */
FGAPI fg_err fg_append_surface_to_chart(fg_chart pChart, fg_surface pSurface);

/**
   Add an existing vector field object to chart

   \param[in] pChart is the chart to which vector field has to be added
   \param[in] pField is the vector field to be added to the chart

   \return \ref fg_err error code
 */
FGAPI fg_err fg_append_vector_field_to_chart(fg_chart pChart, fg_vector_field pField);

/**
   Create and add an Image object to the current chart

   \param[out] pImage is the handle of the image object created
   \param[in] pHandle is chart handle to which image object will be added.
   \param[in] pWidth Width of the image
   \param[in] pHeight Height of the image
   \param[in] pFormat Color channel format of image, uses one of the values
              of \ref fg_channel_format
   \param[in] pType takes one of the values of \ref fg_dtype that indicates
              the integral data type of histogram data

   \return \ref fg_err error code
 */
FGAPI fg_err fg_add_image_to_chart(fg_image* pImage, fg_chart pHandle,
                                   const unsigned pWidth, const unsigned pHeight,
                                   const fg_channel_format pFormat,
                                   const fg_dtype pType);

/**
   Create and add an Histogram object to the current chart

   \param[out] pHistogram is the handle of the histogram object created
   \param[in] pHandle is chart handle
   \param[in] pNBins is number of bins the data is sorted out
   \param[in] pType takes one of the values of \ref fg_dtype that indicates
              the integral data type of histogram data

   \return \ref fg_err error code
 */
FGAPI fg_err fg_add_histogram_to_chart(fg_histogram* pHistogram, fg_chart pHandle,
                                       const unsigned pNBins, const fg_dtype pType);

/**
   Create and add an Plot object to the current chart

   \param[out] pPlot is the handle of the plot object created
   \param[in] pHandle is chart handle
   \param[in] pNPoints is number of data points to display
   \param[in] pType takes one of the values of \ref fg_dtype that indicates
              the integral data type of plot data
   \param[in] pPlotType dictates the type of plot/graph,
              it can take one of the values of \ref fg_plot_type
   \param[in] pMarkerType indicates which symbol is rendered as marker. It can take one of
              the values of \ref fg_marker_type.

   \return \ref fg_err error code
 */
FGAPI fg_err fg_add_plot_to_chart(fg_plot* pPlot, fg_chart pHandle,
                                  const unsigned pNPoints, const fg_dtype pType,
                                  const fg_plot_type pPlotType, const fg_marker_type pMarkerType);

/**
   Create and add an Plot object to the current chart

   \param[out] pSurface is the handle of the surface object created
   \param[in] pHandle is chart handle
   \param[in] pXPoints is number of data points along X dimension
   \param[in] pYPoints is number of data points along Y dimension
   \param[in] pType takes one of the values of \ref fg_dtype that indicates
              the integral data type of plot data
   \param[in] pPlotType is the render type which can be one of \ref fg_plot_type (valid choices
              are FG_PLOT_SURFACE and FG_PLOT_SCATTER)
   \param[in] pMarkerType is the type of \ref fg_marker_type to draw for \ref FG_PLOT_SCATTER plot type

   \return \ref fg_err error code
 */
FGAPI fg_err fg_add_surface_to_chart(fg_surface* pSurface, fg_chart pHandle,
                                     const unsigned pXPoints, const unsigned pYPoints, const fg_dtype pType,
                                     const fg_plot_type pPlotType, const fg_marker_type pMarkerType);

/**
   Create and add an Vector Field object to the current chart

   \param[out] pField is the handle of the Vector Field object created
   \param[in] pHandle is chart handle
   \param[in] pNPoints is number of data points to display
   \param[in] pType takes one of the values of \ref fg_dtype that indicates the integral data type of vector field data

   \return \ref fg_err error code
 */
FGAPI fg_err fg_add_vector_field_to_chart(fg_vector_field* pField, fg_chart pHandle,
                                          const unsigned pNPoints, const fg_dtype pType);

/**
Remove an Image object from the current chart

\param[in] pHandle is chart handle to which image object will be added.
\param[in] pImage is the handle of the image object to remove

\return \ref fg_err error code
*/
FGAPI fg_err fg_remove_image_from_chart(fg_chart pHandle, fg_image pImage);

/**
Remove a Histogram object from the current chart

\param[in] pHandle is chart handle
\param[in] pHistogram is the handle of the histogram object to remove

\return \ref fg_err error code
*/
FGAPI fg_err fg_remove_histogram_from_chart(fg_chart pHandle,
                                            fg_histogram pHistogram);

/**
Remove a Plot object from the current chart

\param[in] pHandle is chart handle
\param[in] pPlot is the handle of the plot object to remove

\return \ref fg_err error code
*/
FGAPI fg_err fg_remove_plot_from_chart(fg_chart pHandle, fg_plot pPlot);

/**
Remove a Plot object from the current chart

\param[in] pHandle is chart handle
\param[in] pSurface is the handle of the surface object to remove

\return \ref fg_err error code
*/
FGAPI fg_err fg_remove_surface_from_chart(fg_chart pHandle,
                                          fg_surface pSurface);

/**
Remove a Vector Field object from the current chart

\param[in] pHandle is chart handle
\param[in] pField is the handle of the Vector Field object to remove

\return \ref fg_err error code
*/
FGAPI fg_err fg_remove_vector_field_from_chart(fg_chart pHandle,
                                               fg_vector_field pField);

/**
   Render the chart to given window

   \param[in] pWindow is target window to where chart will be rendered
   \param[in] pChart is chart handle
   \param[in] pX is x coordinate of origin of viewport in window coordinates
   \param[in] pY is y coordinate of origin of viewport in window coordinates
   \param[in] pWidth is the width of the viewport
   \param[in] pHeight is the height of the viewport

   \return \ref fg_err error code
 */
FGAPI fg_err fg_render_chart(const fg_window pWindow,
                             const fg_chart pChart,
                             const int pX, const int pY, const int pWidth, const int pHeight);

/**
   Render the type of a chart

   \param[out] pChartType return the type of the chart
   \param[in] pChart is chart handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_chart_type(fg_chart_type *pChartType, const fg_chart pChart);

/** @} */

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

namespace forge
{

/**
   \brief Chart is base canvas where other plottable objects are rendered.

   Charts come in two types:
        - \ref FG_CHART_2D - Two dimensional charts
        - \ref FG_CHART_3D - Three dimensional charts
 */
class Chart {
    private:
        fg_chart mValue;

    public:
        /**
           Creates a Chart object with given dimensional property

           \param[in] cType is chart dimension property
         */
        FGAPI Chart(const ChartType cType);

        /**
           Chart copy constructor
         */
        FGAPI Chart(const Chart& pOther);

        /**
           Chart destructor
         */
        FGAPI ~Chart();

        /**
           Set axes titles for the chart

           \param[in] pX is x-axis title label
           \param[in] pY is y-axis title label
           \param[in] pZ is z-axis title label
         */
        FGAPI void setAxesTitles(const char* pX,
                                 const char* pY,
                                 const char* pZ=NULL);

        /**
           Set axes data ranges

           \param[in] pXmin is x-axis minimum data value
           \param[in] pXmax is x-axis maximum data value
           \param[in] pYmin is y-axis minimum data value
           \param[in] pYmax is y-axis maximum data value
           \param[in] pZmin is z-axis minimum data value
           \param[in] pZmax is z-axis maximum data value
         */
        FGAPI void setAxesLimits(const float pXmin, const float pXmax,
                                 const float pYmin, const float pYmax,
                                 const float pZmin=0, const float pZmax=0);

        /**
           Set the format for display of axes labels

           \param[in] pXFormat sets the display format for numbers of X axis
           \param[in] pYFormat sets the display format for numbers of Y axis
           \param[in] pZFormat sets the display format for numbers of Z axis

           Display format string follows printf style formating for numbers
         */
        FGAPI void setAxesLabelFormat(const char* pXFormat,
                                      const char* pYFormat = "%4.1f",
                                      const char* pZFormat = "%4.1f");

        /**
           Get axes data ranges

           \param[out] pXmin is x-axis minimum data value
           \param[out] pXmax is x-axis maximum data value
           \param[out] pYmin is y-axis minimum data value
           \param[out] pYmax is y-axis maximum data value
           \param[out] pZmin is z-axis minimum data value
           \param[out] pZmax is z-axis maximum data value
         */
        FGAPI void getAxesLimits(float* pXmin, float* pXmax,
                                 float* pYmin, float* pYmax,
                                 float* pZmin = NULL, float* pZmax = NULL);

        /**
           Set legend position for Chart

           \param[in] pX is horizontal position in normalized coordinates
           \param[in] pY is vertical position in normalized coordinates

           \note By normalized coordinates, the range of these coordinates is expected to be [0-1].
           (0,0) is the bottom hand left corner.
         */
        FGAPI void setLegendPosition(const float pX, const float pY);

        /**
           Add an existing Image object to the current chart

           \param[in] pImage is the Image to render on the chart
         */
        FGAPI void add(const Image& pImage);

        /**
           Add an existing Histogram object to the current chart

           \param[in] pHistogram is the Histogram to render on the chart
         */
        FGAPI void add(const Histogram& pHistogram);

        /**
           Add an existing Plot object to the current chart

           \param[in] pPlot is the Plot to render on the chart
         */
        FGAPI void add(const Plot& pPlot);

        /**
           Add an existing Surface object to the current chart

           \param[in] pSurface is the Surface to render on the chart
         */
        FGAPI void add(const Surface& pSurface);

        /**
           Add an existing vector field object to the current chart

           \param[in] pVectorField is the Surface to render on the chart
         */
        FGAPI void add(const VectorField& pVectorField);

        /**
        Remove an existing Image object to the current chart

        \param[in] pImage is the Image to remove from the chart
        */
        FGAPI void remove(const Image &pImage);

        /**
        Remove an existing Histogram object to the current chart

        \param[in] pHistogram is the Histogram to remove from the chart
        */
        FGAPI void remove(const Histogram &pHistogram);

        /**
        Remove an existing Plot object to the current chart

        \param[in] pPlot is the Plot to remove from the chart
        */
        FGAPI void remove(const Plot &pPlot);

        /**
        Remove an existing Surface object to the current chart

        \param[in] pSurface is the Surface to remove from the chart
        */
        FGAPI void remove(const Surface &pSurface);

        /**
        Remove an existing vector field object to the current chart

        \param[in] pVectorField is the Vectore to remove from the chart
        */
        FGAPI void remove(const VectorField &pVectorField);

        /**
           Create and add an Image object to the current chart

           \param[in] pWidth Width of the image
           \param[in] pHeight Height of the image
           \param[in] pFormat Color channel format of image, uses one of the values
                      of \ref ChannelFormat
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of histogram data
         */
        FGAPI Image image(const unsigned pWidth, const unsigned pHeight,
                          const ChannelFormat pFormat=FG_RGBA, const dtype pDataType=f32);

        /**
           Create and add an Histogram object to the current chart

           \param[in] pNBins is number of bins the data is sorted out
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of histogram data
         */
        FGAPI Histogram histogram(const unsigned pNBins, const dtype pDataType);

        /**
           Create and add an Plot object to the current chart

           \param[in] pNumPoints is number of data points to display
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of plot data
           \param[in] pPlotType dictates the type of plot/graph,
                      it can take one of the values of \ref PlotType
           \param[in] pMarkerType indicates which symbol is rendered as marker. It can take one of
                      the values of \ref MarkerType.
         */
        FGAPI Plot plot(const unsigned pNumPoints, const dtype pDataType,
                        const PlotType pPlotType=FG_PLOT_LINE, const MarkerType pMarkerType=FG_MARKER_NONE);

        /**
           Create and add an Plot object to the current chart

           \param[in] pNumXPoints is number of data points along X dimension
           \param[in] pNumYPoints is number of data points along Y dimension
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of plot data
           \param[in] pPlotType is the render type which can be one of \ref PlotType (valid choices
                      are FG_PLOT_SURFACE and FG_PLOT_SCATTER)
           \param[in] pMarkerType is the type of \ref MarkerType to draw for \ref FG_PLOT_SCATTER plot type
         */
        FGAPI Surface surface(const unsigned pNumXPoints, const unsigned pNumYPoints, const dtype pDataType,
                              const PlotType pPlotType=FG_PLOT_SURFACE, const MarkerType pMarkerType=FG_MARKER_NONE);

        /**
           Create and add an Vector Field object to the current chart

           \param[in] pNumPoints is number of data points to display
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of vector field data
         */
        FGAPI VectorField vectorField(const unsigned pNumPoints, const dtype pDataType);

        /**
           Render the chart to given window

           \param[in] pWindow is target window to where chart will be rendered
           \param[in] pX is x coordinate of origin of viewport in window coordinates
           \param[in] pY is y coordinate of origin of viewport in window coordinates
           \param[in] pVPW is the width of the viewport
           \param[in] pVPH is the height of the viewport
         */
        FGAPI void render(const Window& pWindow,
                          const int pX, const int pY, const int pVPW, const int pVPH) const;

        /**
           Get the handle to internal implementation of Chart
         */
        FGAPI fg_chart get() const;

        /**
           Get the type of the chart
         */
        FGAPI ChartType getChartType() const;
};

}

#endif
