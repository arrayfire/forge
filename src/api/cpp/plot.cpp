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
#include <chart_renderables.hpp>

namespace forge
{

Plot::Plot(const unsigned pNumPoints, const dtype pDataType, const ChartType pChartType,
           const PlotType pPlotType, const MarkerType pMarkerType)
{
    try {
        mValue = getHandle(new common::Plot(pNumPoints, pDataType, pPlotType, pMarkerType, pChartType));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Plot::Plot(const Plot& pOther)
{
    try {
        mValue = getHandle(new common::Plot(pOther.get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Plot::~Plot()
{
    delete getPlot(mValue);
}

void Plot::setColor(const Color pColor)
{
    try {
        float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
        float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
        float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
        float a = (((int) pColor       ) & 0xFF ) / 255.f;
        getPlot(mValue)->setColor(r, g, b, a);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Plot::setColor(const float pRed, const float pGreen,
                    const float pBlue, const float pAlpha)
{
    try {
        getPlot(mValue)->setColor(pRed, pGreen, pBlue, pAlpha);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Plot::setLegend(const char* pLegend)
{
    try {
        getPlot(mValue)->setLegend(pLegend);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Plot::setMarkerSize(const float pMarkerSize)
{
    try {
        getPlot(mValue)->setMarkerSize(pMarkerSize);
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Plot::vertices() const
{
    try {
        return getPlot(mValue)->vbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Plot::colors() const
{
    try {
        return getPlot(mValue)->cbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Plot::alphas() const
{
    try {
        return getPlot(mValue)->abo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Plot::radii() const
{
    try {
        return getPlot(mValue)->mbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Plot::verticesSize() const
{
    try {
        return (unsigned)getPlot(mValue)->vboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Plot::colorsSize() const
{
    try {
        return (unsigned)getPlot(mValue)->cboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Plot::alphasSize() const
{
    try {
        return (unsigned)getPlot(mValue)->aboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Plot::radiiSize() const
{
    try {
        return (unsigned)getPlot(mValue)->mboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

fg_plot Plot::get() const
{
    try {
        return mValue;
    } CATCH_INTERNAL_TO_EXTERNAL
}

}
