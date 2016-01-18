/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot.h>
#include <plot.hpp>

#include <cmath>

using namespace std;

namespace fg
{

Plot::Plot(const uint pNumPoints, const dtype pDataType, const ChartType pChartType,
           const PlotType pPlotType, const MarkerType pMarkerType)
{
    mValue = new internal::_Plot(pNumPoints, pDataType, pPlotType, pMarkerType, pChartType);
}

Plot::Plot(const Plot& pOther)
{
    mValue = new internal::_Plot(*pOther.get());
}

Plot::~Plot()
{
    delete mValue;
}

void Plot::setColor(const Color pColor)
{
    float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
    float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
    float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
    float a = (((int) pColor       ) & 0xFF ) / 255.f;
    mValue->setColor(r, g, b, a);
}

void Plot::setColor(const float pRed, const float pGreen,
                    const float pBlue, const float pAlpha)
{
    mValue->setColor(pRed, pGreen, pBlue, pAlpha);
}

void Plot::setLegend(const std::string& pLegend)
{
    mValue->setLegend(pLegend);
}

uint Plot::vertices() const
{
    return mValue->vbo();
}

uint Plot::colors() const
{
    return mValue->cbo();
}

uint Plot::alphas() const
{
    return mValue->abo();
}

uint Plot::verticesSize() const
{
    return (uint)mValue->vboSize();
}

uint Plot::colorsSize() const
{
    return (uint)mValue->cboSize();
}

uint Plot::alphasSize() const
{
    return (uint)mValue->aboSize();
}

internal::_Plot* Plot::get() const
{
    return mValue;
}

}
