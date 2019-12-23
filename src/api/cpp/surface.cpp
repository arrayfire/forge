/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/surface.h>

#include <error.hpp>

#include <utility>

namespace forge {
Surface::Surface(unsigned pNumXPoints, unsigned pNumYPoints, dtype pDataType,
                 PlotType pPlotType, MarkerType pMarkerType) {
    fg_surface temp = 0;
    FG_THROW(fg_create_surface(&temp, pNumXPoints, pNumYPoints,
                               (fg_dtype)pDataType, pPlotType, pMarkerType));
    std::swap(mValue, temp);
}

Surface::Surface(const Surface& other) {
    fg_surface temp = 0;

    FG_THROW(fg_retain_surface(&temp, other.get()));

    std::swap(mValue, temp);
}

Surface::Surface(const fg_surface pHandle) : mValue(pHandle) {}

Surface::~Surface() { fg_release_surface(get()); }

void Surface::setColor(const Color pColor) {
    float r = (((int)pColor >> 24) & 0xFF) / 255.f;
    float g = (((int)pColor >> 16) & 0xFF) / 255.f;
    float b = (((int)pColor >> 8) & 0xFF) / 255.f;
    float a = (((int)pColor) & 0xFF) / 255.f;

    FG_THROW(fg_set_surface_color(get(), r, g, b, a));
}

void Surface::setColor(const float pRed, const float pGreen, const float pBlue,
                       const float pAlpha) {
    FG_THROW(fg_set_surface_color(get(), pRed, pGreen, pBlue, pAlpha));
}

void Surface::setLegend(const char* pLegend) {
    FG_THROW(fg_set_surface_legend(get(), pLegend));
}

unsigned Surface::vertices() const {
    unsigned temp = 0;
    FG_THROW(fg_get_surface_vertex_buffer(&temp, get()));
    return temp;
}

unsigned Surface::colors() const {
    unsigned temp = 0;
    FG_THROW(fg_get_surface_color_buffer(&temp, get()));
    return temp;
}

unsigned Surface::alphas() const {
    unsigned temp = 0;
    FG_THROW(fg_get_surface_alpha_buffer(&temp, get()));
    return temp;
}

unsigned Surface::verticesSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_surface_vertex_buffer_size(&temp, get()));
    return temp;
}

unsigned Surface::colorsSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_surface_color_buffer_size(&temp, get()));
    return temp;
}

unsigned Surface::alphasSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_surface_alpha_buffer_size(&temp, get()));
    return temp;
}

fg_surface Surface::get() const { return mValue; }
}  // namespace forge
