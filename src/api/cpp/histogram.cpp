/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/histogram.h>

#include <handle.hpp>
#include <chart_renderables.hpp>

namespace forge
{

Histogram::Histogram(const uint pNBins, const dtype pDataType)
{
    mValue = getHandle(new common::Histogram(pNBins, pDataType));
}

Histogram::Histogram(const Histogram& pOther)
{
    mValue = getHandle(new common::Histogram(pOther.get()));
}

Histogram::~Histogram()
{
    delete getHistogram(mValue);
}

void Histogram::setColor(const Color pColor)
{
    float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
    float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
    float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
    float a = (((int) pColor       ) & 0xFF ) / 255.f;
    getHistogram(mValue)->setColor(r, g, b, a);
}

void Histogram::setColor(const float pRed, const float pGreen,
                         const float pBlue, const float pAlpha)
{
    getHistogram(mValue)->setColor(pRed, pGreen, pBlue, pAlpha);
}

void Histogram::setLegend(const char* pLegend)
{
    getHistogram(mValue)->setLegend(pLegend);
}

uint Histogram::vertices() const
{
    return getHistogram(mValue)->vbo();
}

uint Histogram::colors() const
{
    return getHistogram(mValue)->cbo();
}

uint Histogram::alphas() const
{
    return getHistogram(mValue)->abo();
}

uint Histogram::verticesSize() const
{
    return (uint)getHistogram(mValue)->vboSize();
}

uint Histogram::colorsSize() const
{
    return (uint)getHistogram(mValue)->cboSize();
}

uint Histogram::alphasSize() const
{
    return (uint)getHistogram(mValue)->aboSize();
}

fg_histogram Histogram::get() const
{
    return mValue;
}

}
