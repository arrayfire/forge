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
#include <chart_renderables.hpp>

namespace forge
{

Surface::Surface(unsigned pNumXPoints, unsigned pNumYPoints, dtype pDataType, PlotType pPlotType, MarkerType pMarkerType)
{
    try {
        mValue = getHandle(new common::Surface(pNumXPoints, pNumYPoints, pDataType, pPlotType, pMarkerType));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Surface::Surface(const Surface& other)
{
    try {
        mValue = getHandle(new common::Surface(other.get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Surface::~Surface()
{
    delete getSurface(mValue);
}

void Surface::setColor(const Color pColor)
{
    try {
        float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
        float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
        float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
        float a = (((int) pColor       ) & 0xFF ) / 255.f;
        getSurface(mValue)->setColor(r, g, b, a);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Surface::setColor(const float pRed, const float pGreen,
                    const float pBlue, const float pAlpha)
{
    try {
        getSurface(mValue)->setColor(pRed, pGreen, pBlue, pAlpha);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Surface::setLegend(const char* pLegend)
{
    try {
        getSurface(mValue)->setLegend(pLegend);
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Surface::vertices() const
{
    try {
        return getSurface(mValue)->vbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Surface::colors() const
{
    try {
        return getSurface(mValue)->cbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Surface::alphas() const
{
    try {
        return getSurface(mValue)->abo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Surface::verticesSize() const
{
    try {
        return (unsigned)getSurface(mValue)->vboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Surface::colorsSize() const
{
    try {
        return (unsigned)getSurface(mValue)->cboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Surface::alphasSize() const
{
    try {
        return (unsigned)getSurface(mValue)->aboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

fg_surface Surface::get() const
{
    try {
        return mValue;
    } CATCH_INTERNAL_TO_EXTERNAL
}

}
