/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/pie.h>

#include <error.hpp>

#include <utility>

namespace forge {
Pie::Pie(const unsigned pNSectors, const dtype pDataType) {
    fg_pie temp = 0;
    FG_THROW(fg_create_pie(&temp, pNSectors, (fg_dtype)pDataType));
    std::swap(mValue, temp);
}

Pie::Pie(const Pie& pOther) {
    fg_pie temp = 0;

    FG_THROW(fg_retain_pie(&temp, pOther.get()));

    std::swap(mValue, temp);
}

Pie::Pie(const fg_pie pHandle) : mValue(pHandle) {}

Pie::~Pie() { fg_release_pie(get()); }

void Pie::setColor(const Color pColor) {
    float r = (((int)pColor >> 24) & 0xFF) / 255.f;
    float g = (((int)pColor >> 16) & 0xFF) / 255.f;
    float b = (((int)pColor >> 8) & 0xFF) / 255.f;
    float a = (((int)pColor) & 0xFF) / 255.f;

    FG_THROW(fg_set_pie_color(get(), r, g, b, a));
}

void Pie::setColor(const float pRed, const float pGreen, const float pBlue,
                   const float pAlpha) {
    FG_THROW(fg_set_pie_color(get(), pRed, pGreen, pBlue, pAlpha));
}

void Pie::setLegend(const char* pLegend) {
    FG_THROW(fg_set_pie_legend(get(), pLegend));
}

unsigned Pie::vertices() const {
    unsigned temp = 0;
    FG_THROW(fg_get_pie_vertex_buffer(&temp, get()));
    return temp;
}

unsigned Pie::colors() const {
    unsigned temp = 0;
    FG_THROW(fg_get_pie_color_buffer(&temp, get()));
    return temp;
}

unsigned Pie::alphas() const {
    unsigned temp = 0;
    FG_THROW(fg_get_pie_alpha_buffer(&temp, get()));
    return temp;
}

unsigned Pie::verticesSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_pie_vertex_buffer_size(&temp, get()));
    return temp;
}

unsigned Pie::colorsSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_pie_color_buffer_size(&temp, get()));
    return temp;
}

unsigned Pie::alphasSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_pie_alpha_buffer_size(&temp, get()));
    return temp;
}

fg_pie Pie::get() const { return mValue; }
}  // namespace forge
