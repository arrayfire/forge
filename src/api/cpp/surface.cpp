/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/surface.h>

#include <handle.hpp>
#include <Surface.hpp>

namespace fg
{

Surface::Surface(unsigned pNumXPoints, unsigned pNumYPoints, dtype pDataType, PlotType pPlotType, MarkerType pMarkerType)
{
    mValue = getHandle(new common::Surface(pNumXPoints, pNumYPoints, pDataType, pPlotType, pMarkerType));
}

Surface::Surface(const Surface& other)
{
    mValue = getHandle(new common::Surface(other.get()));
}

Surface::~Surface()
{
    delete getSurface(mValue);
}

void Surface::setColor(const Color pColor)
{
    float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
    float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
    float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
    float a = (((int) pColor       ) & 0xFF ) / 255.f;
    getSurface(mValue)->setColor(r, g, b, a);
}

void Surface::setColor(const float pRed, const float pGreen,
                    const float pBlue, const float pAlpha)
{
    getSurface(mValue)->setColor(pRed, pGreen, pBlue, pAlpha);
}

void Surface::setLegend(const char* pLegend)
{
    getSurface(mValue)->setLegend(pLegend);
}

uint Surface::vertices() const
{
    return getSurface(mValue)->vbo();
}

uint Surface::colors() const
{
    return getSurface(mValue)->cbo();
}

uint Surface::alphas() const
{
    return getSurface(mValue)->abo();
}

uint Surface::verticesSize() const
{
    return (uint)getSurface(mValue)->vboSize();
}

uint Surface::colorsSize() const
{
    return (uint)getSurface(mValue)->cboSize();
}

uint Surface::alphasSize() const
{
    return (uint)getSurface(mValue)->aboSize();
}

fg_surface Surface::get() const
{
    return mValue;
}

}
