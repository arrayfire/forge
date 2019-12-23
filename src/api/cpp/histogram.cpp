/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/histogram.h>

#include <error.hpp>

#include <utility>

namespace forge {
Histogram::Histogram(const unsigned pNBins, const dtype pDataType) {
    fg_histogram temp = 0;
    FG_THROW(fg_create_histogram(&temp, pNBins, (fg_dtype)pDataType));
    std::swap(mValue, temp);
}

Histogram::Histogram(const Histogram& pOther) {
    fg_histogram temp = 0;

    FG_THROW(fg_retain_histogram(&temp, pOther.get()));

    std::swap(mValue, temp);
}

Histogram::Histogram(const fg_histogram pHandle) : mValue(pHandle) {}

Histogram::~Histogram() { fg_release_histogram(get()); }

void Histogram::setColor(const Color pColor) {
    float r = (((int)pColor >> 24) & 0xFF) / 255.f;
    float g = (((int)pColor >> 16) & 0xFF) / 255.f;
    float b = (((int)pColor >> 8) & 0xFF) / 255.f;
    float a = (((int)pColor) & 0xFF) / 255.f;

    FG_THROW(fg_set_histogram_color(get(), r, g, b, a));
}

void Histogram::setColor(const float pRed, const float pGreen,
                         const float pBlue, const float pAlpha) {
    FG_THROW(fg_set_histogram_color(get(), pRed, pGreen, pBlue, pAlpha));
}

void Histogram::setLegend(const char* pLegend) {
    FG_THROW(fg_set_histogram_legend(get(), pLegend));
}

unsigned Histogram::vertices() const {
    unsigned temp = 0;
    FG_THROW(fg_get_histogram_vertex_buffer(&temp, get()));
    return temp;
}

unsigned Histogram::colors() const {
    unsigned temp = 0;
    FG_THROW(fg_get_histogram_color_buffer(&temp, get()));
    return temp;
}

unsigned Histogram::alphas() const {
    unsigned temp = 0;
    FG_THROW(fg_get_histogram_alpha_buffer(&temp, get()));
    return temp;
}

unsigned Histogram::verticesSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_histogram_vertex_buffer_size(&temp, get()));
    return temp;
}

unsigned Histogram::colorsSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_histogram_color_buffer_size(&temp, get()));
    return temp;
}

unsigned Histogram::alphasSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_histogram_alpha_buffer_size(&temp, get()));
    return temp;
}

fg_histogram Histogram::get() const { return mValue; }
}  // namespace forge
