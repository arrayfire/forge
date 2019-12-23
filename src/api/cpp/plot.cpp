/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot.h>

#include <error.hpp>

#include <utility>

namespace forge {
Plot::Plot(const unsigned pNumPoints, const dtype pDataType,
           const ChartType pChartType, const PlotType pPlotType,
           const MarkerType pMarkerType) {
    fg_plot temp = 0;
    FG_THROW(fg_create_plot(&temp, pNumPoints, (fg_dtype)pDataType, pChartType,
                            pPlotType, pMarkerType));
    std::swap(mValue, temp);
}

Plot::Plot(const Plot& pOther) {
    fg_plot temp = 0;

    FG_THROW(fg_retain_plot(&temp, pOther.get()));

    std::swap(mValue, temp);
}

Plot::Plot(const fg_plot pHandle) : mValue(pHandle) {}

Plot::~Plot() { fg_release_plot(get()); }

void Plot::setColor(const Color pColor) {
    float r = (((int)pColor >> 24) & 0xFF) / 255.f;
    float g = (((int)pColor >> 16) & 0xFF) / 255.f;
    float b = (((int)pColor >> 8) & 0xFF) / 255.f;
    float a = (((int)pColor) & 0xFF) / 255.f;

    FG_THROW(fg_set_plot_color(get(), r, g, b, a));
}

void Plot::setColor(const float pRed, const float pGreen, const float pBlue,
                    const float pAlpha) {
    FG_THROW(fg_set_plot_color(get(), pRed, pGreen, pBlue, pAlpha));
}

void Plot::setLegend(const char* pLegend) {
    FG_THROW(fg_set_plot_legend(get(), pLegend));
}

void Plot::setMarkerSize(const float pMarkerSize) {
    FG_THROW(fg_set_plot_marker_size(get(), pMarkerSize));
}

unsigned Plot::vertices() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_vertex_buffer(&temp, get()));
    return temp;
}

unsigned Plot::colors() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_color_buffer(&temp, get()));
    return temp;
}

unsigned Plot::alphas() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_alpha_buffer(&temp, get()));
    return temp;
}

unsigned Plot::radii() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_radii_buffer(&temp, get()));
    return temp;
}

unsigned Plot::verticesSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_vertex_buffer_size(&temp, get()));
    return temp;
}

unsigned Plot::colorsSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_color_buffer_size(&temp, get()));
    return temp;
}

unsigned Plot::alphasSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_alpha_buffer_size(&temp, get()));
    return temp;
}

unsigned Plot::radiiSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_radii_buffer_size(&temp, get()));
    return temp;
}

fg_plot Plot::get() const { return mValue; }
}  // namespace forge
