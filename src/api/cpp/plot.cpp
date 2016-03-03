/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot.h>

#include <handle.hpp>
#include <Plot.hpp>

namespace fg
{

Plot::Plot(const uint pNumPoints, const dtype pDataType, const ChartType pChartType,
           const PlotType pPlotType, const MarkerType pMarkerType)
{
    mValue = getHandle(new common::Plot(pNumPoints, pDataType, pPlotType, pMarkerType, pChartType));
}

Plot::Plot(const Plot& pOther)
{
    mValue = getHandle(new common::Plot(*getPlot(pOther.get())));
}

Plot::~Plot()
{
    delete getPlot(mValue);
}

void Plot::setColor(const Color pColor)
{
    float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
    float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
    float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
    float a = (((int) pColor       ) & 0xFF ) / 255.f;
    getPlot(mValue)->setColor(r, g, b, a);
}

void Plot::setColor(const float pRed, const float pGreen,
                    const float pBlue, const float pAlpha)
{
    getPlot(mValue)->setColor(pRed, pGreen, pBlue, pAlpha);
}

void Plot::setLegend(const char* pLegend)
{
    getPlot(mValue)->setLegend(pLegend);
}

void Plot::setMarkerSize(const float pMarkerSize)
{
    getPlot(mValue)->setMarkerSize(pMarkerSize);
}

uint Plot::vertices() const
{
    return getPlot(mValue)->vbo();
}

uint Plot::colors() const
{
    return getPlot(mValue)->cbo();
}

uint Plot::alphas() const
{
    return getPlot(mValue)->abo();
}

uint Plot::markers() const
{
    return getPlot(mValue)->mbo();
}

uint Plot::verticesSize() const
{
    return (uint)getPlot(mValue)->vboSize();
}

uint Plot::colorsSize() const
{
    return (uint)getPlot(mValue)->cboSize();
}

uint Plot::alphasSize() const
{
    return (uint)getPlot(mValue)->aboSize();
}

uint Plot::markersSize() const
{
    return (uint)getPlot(mValue)->mboSize();
}

fg_plot Plot::get() const
{
    return mValue;
}

}
