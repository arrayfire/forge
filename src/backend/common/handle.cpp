/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_handling.hpp>
#include <common/handle.hpp>

namespace forge {
namespace common {

fg_window getHandle(Window* pValue) {
    return reinterpret_cast<fg_window>(pValue);
}

fg_font getHandle(Font* pValue) { return reinterpret_cast<fg_font>(pValue); }

fg_image getHandle(Image* pValue) { return reinterpret_cast<fg_image>(pValue); }

fg_chart getHandle(Chart* pValue) { return reinterpret_cast<fg_chart>(pValue); }

fg_histogram getHandle(Histogram* pValue) {
    return reinterpret_cast<fg_histogram>(pValue);
}

fg_plot getHandle(Plot* pValue) { return reinterpret_cast<fg_plot>(pValue); }

fg_surface getHandle(Surface* pValue) {
    return reinterpret_cast<fg_surface>(pValue);
}

fg_vector_field getHandle(VectorField* pValue) {
    return reinterpret_cast<fg_vector_field>(pValue);
}

Window* getWindow(const fg_window& pValue) {
    return reinterpret_cast<common::Window*>(pValue);
}

Font* getFont(const fg_font& pValue) {
    return reinterpret_cast<common::Font*>(pValue);
}

Image* getImage(const fg_image& pValue) {
    return reinterpret_cast<common::Image*>(pValue);
}

Chart* getChart(const fg_chart& pValue) {
    return reinterpret_cast<common::Chart*>(pValue);
}

Histogram* getHistogram(const fg_histogram& pValue) {
    return reinterpret_cast<common::Histogram*>(pValue);
}

Plot* getPlot(const fg_plot& pValue) {
    return reinterpret_cast<common::Plot*>(pValue);
}

Surface* getSurface(const fg_surface& pValue) {
    return reinterpret_cast<common::Surface*>(pValue);
}

VectorField* getVectorField(const fg_vector_field& pValue) {
    return reinterpret_cast<common::VectorField*>(pValue);
}

}  // namespace common
}  // namespace forge
