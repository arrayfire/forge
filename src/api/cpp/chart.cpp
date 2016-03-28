/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/chart.h>
#include <fg/font.h>
#include <fg/histogram.h>
#include <fg/image.h>
#include <fg/plot.h>
#include <fg/surface.h>
#include <fg/window.h>

#include <handle.hpp>
#include <chart.hpp>
#include <chart_renderables.hpp>
#include <font.hpp>
#include <image.hpp>
#include <window.hpp>

namespace fg
{

Chart::Chart(const ChartType cType)
{
    mValue = getHandle(new common::Chart(cType));
}

Chart::Chart(const Chart& pOther)
{
    mValue = getHandle(new common::Chart(pOther.get()));
}

Chart::~Chart()
{
    delete getChart(mValue);
}

void Chart::setAxesTitles(const char* pX,
                          const char* pY,
                          const char* pZ)
{
    getChart(mValue)->setAxesTitles(pX, pY, pZ);
}

void Chart::setAxesLimits(const float pXmin, const float pXmax,
                          const float pYmin, const float pYmax,
                          const float pZmin, const float pZmax)
{
    getChart(mValue)->setAxesLimits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax);
}

void Chart::setLegendPosition(const float pX, const float pY)
{
    getChart(mValue)->setLegendPosition(pX, pY);
}

void Chart::add(const Image& pImage)
{
    getChart(mValue)->addRenderable(getImage(pImage.get())->impl());
}

void Chart::add(const Histogram& pHistogram)
{
    getChart(mValue)->addRenderable(getHistogram(pHistogram.get())->impl());
}

void Chart::add(const Plot& pPlot)
{
    getChart(mValue)->addRenderable(getPlot(pPlot.get())->impl());
}

void Chart::add(const Surface& pSurface)
{
    getChart(mValue)->addRenderable(getSurface(pSurface.get())->impl());
}

Image Chart::image(const uint pWidth, const uint pHeight,
                   const ChannelFormat pFormat, const dtype pDataType)
{
    Image retVal(pWidth, pHeight, pFormat, pDataType);
    getChart(mValue)->addRenderable(getImage(retVal.get())->impl());
    return retVal;
}

Histogram Chart::histogram(const uint pNBins, const dtype pDataType)
{
    common::Chart* chrt = getChart(mValue);
    ChartType ctype = chrt->chartType();

    if (ctype == FG_CHART_2D) {
        Histogram retVal(pNBins, pDataType);
        chrt->addRenderable(getHistogram(retVal.get())->impl());
        return retVal;
    } else {
        throw ArgumentError("Chart::render", __LINE__, 5,
                "Can add histogram to a 2d chart only");
    }
}

Plot Chart::plot(const uint pNumPoints, const dtype pDataType,
                 const PlotType pPlotType, const MarkerType pMarkerType)
{
    common::Chart* chrt = getChart(mValue);
    ChartType ctype = chrt->chartType();

    if (ctype == FG_CHART_2D) {
        Plot retVal(pNumPoints, pDataType, FG_CHART_2D, pPlotType, pMarkerType);
        chrt->addRenderable(getPlot(retVal.get())->impl());
        return retVal;
    } else {
        Plot retVal(pNumPoints, pDataType, FG_CHART_3D, pPlotType, pMarkerType);
        chrt->addRenderable(getPlot(retVal.get())->impl());
        return retVal;
    }
}

Surface Chart::surface(const uint pNumXPoints, const uint pNumYPoints, const dtype pDataType,
                       const PlotType pPlotType, const MarkerType pMarkerType)
{
    common::Chart* chrt = getChart(mValue);
    ChartType ctype = chrt->chartType();

    if (ctype == FG_CHART_3D) {
        Surface retVal(pNumXPoints, pNumYPoints, pDataType, pPlotType, pMarkerType);
        getChart(mValue)->addRenderable(getSurface(retVal.get())->impl());
        return retVal;
    } else {
        throw ArgumentError("Chart::render", __LINE__, 5,
                "Can add surface plot to a 3d chart only");
    }
}

void Chart::render(const Window& pWindow,
                   const int pX, const int pY, const int pVPW, const int pVPH,
                   const float* pTransform) const
{
    getChart(mValue)->render(getWindow(pWindow.get())->getID(),
                             pX, pY, pVPW, pVPH,
                             glm::make_mat4(pTransform));
}

fg_chart Chart::get() const
{
    return getChart(mValue);
}

}
