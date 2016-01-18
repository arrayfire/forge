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

#include <string>
#include <vector>

namespace internal
{
class _Chart;
}

namespace fg
{

/**
   \class Chart
 */
class Chart {
    private:
        ChartType mChartType;
        internal::_Chart* mValue;

    public:
        /**
           Creates a Chart object with given dimensional property

           \param[in] pType is chart dimension property
         */
        FGAPI Chart(const ChartType cType);

        /**
           Chart destructor
         */
        FGAPI ~Chart();

        /**
           Set axes titles for the chart

           \param[in] x is x-axis title label
           \param[in] y is y-axis title label
           \param[in] z is z-axis title label
         */
        FGAPI void setAxesTitles(const std::string pX,
                                 const std::string pY,
                                 const std::string pZ=std::string(""));

        /**
           Set axes data ranges

           \param[in] xmin is x-axis minimum data value
           \param[in] xmax is x-axis maximum data value
           \param[in] ymin is y-axis minimum data value
           \param[in] ymax is y-axis maximum data value
           \param[in] zmin is z-axis minimum data value
           \param[in] zmax is z-axis maximum data value
         */
        FGAPI void setAxesLimits(const float pXmin, const float pXmax,
                                 const float pYmin, const float pYmax,
                                 const float pZmin=-1, const float pZmax=1);

        FGAPI void add(const Image& pImage);
        FGAPI void add(const Histogram& pHistogram);
        FGAPI void add(const Plot& pPlot);
        FGAPI void add(const Surface& pSurface);

        FGAPI Image image(const uint pWidth, const uint pHeight,
                          const ChannelFormat pFormat=FG_RGBA, const dtype pDataType=f32);

        FGAPI Histogram histogram(const uint pNBins, const dtype pDataType);

        FGAPI Plot plot(const uint pNumPoints, const dtype pDataType,
                        const PlotType pPlotType=FG_LINE, const MarkerType pMarkerType=FG_NONE);

        FGAPI Surface surface(const uint pNumXPoints, const uint pNumYPoints, const dtype pDataType,
                              const PlotType pPlotType=FG_SURFACE, const MarkerType pMarkerType=FG_NONE);

        /**
           Render the chart to given window

           \param[in] pWindow is target window to where chart will be rendered
           \param[in] pX is x coordinate of origin of viewport in window coordinates
           \param[in] pY is y coordinate of origin of viewport in window coordinates
           \param[in] pVPW is the width of the viewport
           \param[in] pVPH is the height of the viewport
           \param[in] pTransform is an array of floats. This vector is expected to contain
                      at least 16 elements
         */
        FGAPI void render(const Window& pWindow,
                          const int pX, const int pY, const int pVPW, const int pVPH,
                          const std::vector<float>& pTransform) const;

        /**
           Get the handle to internal implementation of Chart
         */
        FGAPI internal::_Chart* get() const;
};

}
