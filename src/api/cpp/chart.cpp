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

namespace forge
{

Chart::Chart(const ChartType cType)
{
    try {
        mValue = getHandle(new common::Chart(cType));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Chart::Chart(const Chart& pOther)
{
    try {
        mValue = getHandle(new common::Chart(pOther.get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Chart::~Chart()
{
    try {
        delete getChart(mValue);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Chart::setAxesTitles(const char* pX,
                          const char* pY,
                          const char* pZ)
{
    try {
        getChart(mValue)->setAxesTitles(pX, pY, pZ);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Chart::setAxesLimits(const float pXmin, const float pXmax,
                          const float pYmin, const float pYmax,
                          const float pZmin, const float pZmax)
{
    try {
        getChart(mValue)->setAxesLimits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Chart::setLegendPosition(const float pX, const float pY)
{
    try {
        getChart(mValue)->setLegendPosition(pX, pY);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Chart::add(const Image& pImage)
{
    try {
        getChart(mValue)->addRenderable(getImage(pImage.get())->impl());
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Chart::add(const Histogram& pHistogram)
{
    try {
        getChart(mValue)->addRenderable(getHistogram(pHistogram.get())->impl());
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Chart::add(const Plot& pPlot)
{
    try {
        getChart(mValue)->addRenderable(getPlot(pPlot.get())->impl());
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Chart::add(const Surface& pSurface)
{
    try {
        getChart(mValue)->addRenderable(getSurface(pSurface.get())->impl());
    } CATCH_INTERNAL_TO_EXTERNAL
}

Image Chart::image(const uint pWidth, const uint pHeight,
                   const ChannelFormat pFormat, const dtype pDataType)
{
    try {
        Image retVal = Image(pWidth, pHeight, pFormat, pDataType);
        getChart(mValue)->addRenderable(getImage(retVal.get())->impl());
        return retVal;
    } CATCH_INTERNAL_TO_EXTERNAL
}

Histogram Chart::histogram(const uint pNBins, const dtype pDataType)
{
    try {
        common::Chart* chrt = getChart(mValue);
        ChartType ctype = chrt->chartType();

        // Histogram is allowed only in FG_CHART_2D
        ARG_ASSERT(5, ctype == FG_CHART_2D);

        Histogram retVal = Histogram(pNBins, pDataType);
        chrt->addRenderable(getHistogram(retVal.get())->impl());
        return retVal;
    } CATCH_INTERNAL_TO_EXTERNAL
}

Plot Chart::plot(const uint pNumPoints, const dtype pDataType,
                 const PlotType pPlotType, const MarkerType pMarkerType)
{
    try {
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
    } CATCH_INTERNAL_TO_EXTERNAL
}

Surface Chart::surface(const uint pNumXPoints, const uint pNumYPoints, const dtype pDataType,
                       const PlotType pPlotType, const MarkerType pMarkerType)
{
    try {
        common::Chart* chrt = getChart(mValue);
        ChartType ctype = chrt->chartType();

        // Surface is allowed only in FG_CHART_3D
        ARG_ASSERT(5, ctype == FG_CHART_3D);

        Surface retVal(pNumXPoints, pNumYPoints, pDataType, pPlotType, pMarkerType);
        getChart(mValue)->addRenderable(getSurface(retVal.get())->impl());
        return retVal;
    } CATCH_INTERNAL_TO_EXTERNAL
}

VectorField Chart::vectorField(const uint pNumPoints, const dtype pDataType)
{
    try {
        common::Chart* chrt = getChart(mValue);
        VectorField retVal(pNumPoints, pDataType, chrt->chartType());
        chrt->addRenderable(getVectorField(retVal.get())->impl());
        return retVal;
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Chart::render(const Window& pWindow,
                   const int pX, const int pY, const int pVPW, const int pVPH) const
{
    try {
        getChart(mValue)->render(getWindow(pWindow.get())->getID(),
                                 pX, pY, pVPW, pVPH,
                                 IDENTITY, IDENTITY);
    } CATCH_INTERNAL_TO_EXTERNAL
}

fg_chart Chart::get() const
{
    try {
        return getChart(mValue);
    } CATCH_INTERNAL_TO_EXTERNAL
}

}
