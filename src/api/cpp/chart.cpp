/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <error.hpp>
#include <fg/chart.h>
#include <fg/font.h>
#include <fg/histogram.h>
#include <fg/image.h>
#include <fg/plot.h>
#include <fg/surface.h>
#include <fg/window.h>

#include <utility>

namespace forge {

Chart::Chart(const ChartType cType) {
    fg_chart temp = 0;
    FG_THROW(fg_create_chart(&temp, (fg_chart_type)cType));
    std::swap(mValue, temp);
}

Chart::Chart(const Chart& pOther) {
    fg_chart temp = 0;
    FG_THROW(fg_retain_chart(&temp, pOther.get()));
    std::swap(mValue, temp);
}

Chart::~Chart() { fg_release_chart(get()); }

void Chart::setAxesTitles(const char* pX, const char* pY, const char* pZ) {
    FG_THROW(fg_set_chart_axes_titles(get(), pX, pY, pZ));
}

void Chart::setAxesLimits(const float pXmin, const float pXmax,
                          const float pYmin, const float pYmax,
                          const float pZmin, const float pZmax) {
    FG_THROW(fg_set_chart_axes_limits(get(), pXmin, pXmax, pYmin, pYmax, pZmin,
                                      pZmax));
}

void Chart::setAxesLabelFormat(const char* pXFormat, const char* pYFormat,
                               const char* pZFormat) {
    FG_THROW(fg_set_chart_label_format(get(), pXFormat, pYFormat, pZFormat));
}

void Chart::getAxesLimits(float* pXmin, float* pXmax, float* pYmin,
                          float* pYmax, float* pZmin, float* pZmax) {
    FG_THROW(fg_get_chart_axes_limits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax,
                                      get()));
}

void Chart::setLegendPosition(const float pX, const float pY) {
    FG_THROW(fg_set_chart_legend_position(get(), pX, pY));
}

void Chart::add(const Image& pImage) {
    FG_THROW(fg_append_image_to_chart(get(), pImage.get()));
}

void Chart::add(const Histogram& pHistogram) {
    FG_THROW(fg_append_histogram_to_chart(get(), pHistogram.get()));
}

void Chart::add(const Plot& pPlot) {
    FG_THROW(fg_append_plot_to_chart(get(), pPlot.get()));
}

void Chart::add(const Surface& pSurface) {
    FG_THROW(fg_append_surface_to_chart(get(), pSurface.get()));
}

void Chart::add(const VectorField& pVectorField) {
    FG_THROW(fg_append_vector_field_to_chart(get(), pVectorField.get()));
}

Image Chart::image(const unsigned pWidth, const unsigned pHeight,
                   const ChannelFormat pFormat, const dtype pDataType) {
    fg_image temp = 0;
    FG_THROW(fg_add_image_to_chart(&temp, get(), pWidth, pHeight,
                                   (fg_channel_format)pFormat,
                                   (fg_dtype)pDataType));
    return Image(temp);
}

Histogram Chart::histogram(const unsigned pNBins, const dtype pDataType) {
    fg_histogram temp = 0;
    FG_THROW(
        fg_add_histogram_to_chart(&temp, get(), pNBins, (fg_dtype)pDataType));
    return Histogram(temp);
}

Plot Chart::plot(const unsigned pNumPoints, const dtype pDataType,
                 const PlotType pPlotType, const MarkerType pMarkerType) {
    fg_plot temp = 0;
    FG_THROW(fg_add_plot_to_chart(&temp, get(), pNumPoints, (fg_dtype)pDataType,
                                  pPlotType, pMarkerType));
    return Plot(temp);
}

Surface Chart::surface(const unsigned pNumXPoints, const unsigned pNumYPoints,
                       const dtype pDataType, const PlotType pPlotType,
                       const MarkerType pMarkerType) {
    fg_surface temp = 0;
    FG_THROW(fg_add_surface_to_chart(&temp, get(), pNumXPoints, pNumYPoints,
                                     (fg_dtype)pDataType, pPlotType,
                                     pMarkerType));
    return Surface(temp);
}

VectorField Chart::vectorField(const unsigned pNumPoints,
                               const dtype pDataType) {
    fg_vector_field temp = 0;
    FG_THROW(fg_add_vector_field_to_chart(&temp, get(), pNumPoints,
                                          (fg_dtype)pDataType));
    return VectorField(temp);
}

void Chart::render(const Window& pWindow, const int pX, const int pY,
                   const int pVPW, const int pVPH) const {
    FG_THROW(fg_render_chart(pWindow.get(), get(), pX, pY, pVPW, pVPH));
}

fg_chart Chart::get() const { return mValue; }

ChartType Chart::getChartType() const {
    fg_chart_type retVal = (fg_chart_type)0;
    FG_THROW(fg_get_chart_type(&retVal, get()));
    return retVal;
}

}  // namespace forge
