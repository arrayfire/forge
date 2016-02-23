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
#include <fg/histogram.h>

namespace internal
{
class _Chart;
}

namespace fg
{

/**
   \class Chart

   \brief Chart is base canvas where other plottable objects are rendered.

   Charts come in two types:
        - \ref FG_2D - Two dimensional charts
        - \ref FG_3D - Three dimensional charts
 */
class Chart {
    private:
        ChartType mChartType;
        internal::_Chart* mValue;

    public:
        /**
           Creates a Chart object with given dimensional property

           \param[in] cType is chart dimension property
         */
        FGAPI Chart(const ChartType cType);

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
                                 const float pZmin=-1, const float pZmax=1);

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
           Create and add an Image object to the current chart

           \param[in] pWidth Width of the image
           \param[in] pHeight Height of the image
           \param[in] pFormat Color channel format of image, uses one of the values
                      of \ref ChannelFormat
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of histogram data
         */
        FGAPI Image image(const uint pWidth, const uint pHeight,
                          const ChannelFormat pFormat=FG_RGBA, const dtype pDataType=f32);

        /**
           Create and add an Histogram object to the current chart

           \param[in] pNBins is number of bins the data is sorted out
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of histogram data
         */
        FGAPI Histogram histogram(const uint pNBins, const dtype pDataType);

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
        FGAPI Plot plot(const uint pNumPoints, const dtype pDataType,
                        const PlotType pPlotType=FG_LINE, const MarkerType pMarkerType=FG_NONE);

        /**
           Create and add an Plot object to the current chart

           \param[in] pNumXPoints is number of data points along X dimension
           \param[in] pNumYPoints is number of data points along Y dimension
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of plot data
           \param[in] pPlotType is the render type which can be one of \ref PlotType (valid choices
                      are FG_SURFACE and FG_SCATTER)
           \param[in] pMarkerType is the type of \ref MarkerType to draw for \ref FG_SCATTER plot type
         */
        FGAPI Surface surface(const uint pNumXPoints, const uint pNumYPoints, const dtype pDataType,
                              const PlotType pPlotType=FG_SURFACE, const MarkerType pMarkerType=FG_NONE);

        /**
           Render the chart to given window

           \param[in] pWindow is target window to where chart will be rendered
           \param[in] pX is x coordinate of origin of viewport in window coordinates
           \param[in] pY is y coordinate of origin of viewport in window coordinates
           \param[in] pVPW is the width of the viewport
           \param[in] pVPH is the height of the viewport
           \param[in] pTransform is an array of floats. This array is expected to contain
                      at least 16 elements

           Note: pTransform array is assumed to be of expected length.
         */
        FGAPI void render(const Window& pWindow,
                          const int pX, const int pY, const int pVPW, const int pVPH,
                          const float* pTransform) const;

        /**
           Get the handle to internal implementation of Chart
         */
        FGAPI internal::_Chart* get() const;
};

}
