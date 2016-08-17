/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/vector_field.h>

#include <handle.hpp>
#include <chart_renderables.hpp>

namespace forge
{

VectorField::VectorField(const uint pNumPoints, const dtype pDataType, const ChartType pChartType)
{
    mValue = getHandle(new common::VectorField(pNumPoints, pDataType, pChartType));
}

VectorField::VectorField(const VectorField& pOther)
{
    mValue = getHandle(new common::VectorField(pOther.get()));
}

VectorField::~VectorField()
{
    delete getVectorField(mValue);
}

void VectorField::setColor(const Color pColor)
{
    float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
    float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
    float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
    float a = (((int) pColor       ) & 0xFF ) / 255.f;
    getVectorField(mValue)->setColor(r, g, b, a);
}

void VectorField::setColor(const float pRed, const float pGreen,
                           const float pBlue, const float pAlpha)
{
    getVectorField(mValue)->setColor(pRed, pGreen, pBlue, pAlpha);
}

void VectorField::setLegend(const char* pLegend)
{
    getVectorField(mValue)->setLegend(pLegend);
}

uint VectorField::vertices() const
{
    return getVectorField(mValue)->vbo();
}

uint VectorField::colors() const
{
    return getVectorField(mValue)->cbo();
}

uint VectorField::alphas() const
{
    return getVectorField(mValue)->abo();
}

uint VectorField::directions() const
{
    return getVectorField(mValue)->dbo();
}

uint VectorField::verticesSize() const
{
    return (uint)getVectorField(mValue)->vboSize();
}

uint VectorField::colorsSize() const
{
    return (uint)getVectorField(mValue)->cboSize();
}

uint VectorField::alphasSize() const
{
    return (uint)getVectorField(mValue)->aboSize();
}

uint VectorField::directionsSize() const
{
    return (uint)getVectorField(mValue)->dboSize();
}

fg_vector_field VectorField::get() const
{
    return mValue;
}

}
