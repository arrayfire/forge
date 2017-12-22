/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <fg/exception.h>

#include <window.hpp>
#include <font.hpp>
#include <image.hpp>
#include <chart.hpp>
#include <chart_renderables.hpp>

fg_window getHandle(forge::common::Window* pValue);

fg_font getHandle(forge::common::Font* pValue);

fg_image getHandle(forge::common::Image* pValue);

fg_chart getHandle(forge::common::Chart* pValue);

fg_histogram getHandle(forge::common::Histogram* pValue);

fg_plot getHandle(forge::common::Plot* pValue);

fg_surface getHandle(forge::common::Surface* pValue);

fg_vector_field getHandle(forge::common::VectorField* pValue);

forge::common::Window* getWindow(const fg_window& pValue);

forge::common::Font* getFont(const fg_font& pValue);

forge::common::Image* getImage(const fg_image& pValue);

forge::common::Chart* getChart(const fg_chart& pValue);

forge::common::Histogram* getHistogram(const fg_histogram& pValue);

forge::common::Plot* getPlot(const fg_plot& pValue);

forge::common::Surface* getSurface(const fg_surface& pValue);

forge::common::VectorField* getVectorField(const fg_vector_field& pValue);
