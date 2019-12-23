/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/chart.hpp>
#include <common/chart_renderables.hpp>
#include <common/font.hpp>
#include <common/image.hpp>
#include <common/window.hpp>
#include <fg/exception.h>

namespace forge {
namespace common {

fg_window getHandle(Window* pValue);

fg_font getHandle(Font* pValue);

fg_image getHandle(Image* pValue);

fg_chart getHandle(Chart* pValue);

fg_histogram getHandle(Histogram* pValue);

fg_plot getHandle(Plot* pValue);

fg_surface getHandle(Surface* pValue);

fg_vector_field getHandle(VectorField* pValue);

Window* getWindow(const fg_window& pValue);

Font* getFont(const fg_font& pValue);

Image* getImage(const fg_image& pValue);

Chart* getChart(const fg_chart& pValue);

Histogram* getHistogram(const fg_histogram& pValue);

Plot* getPlot(const fg_plot& pValue);

Surface* getSurface(const fg_surface& pValue);

VectorField* getVectorField(const fg_vector_field& pValue);

}  // namespace common
}  // namespace forge
