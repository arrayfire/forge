/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <handle.hpp>

fg_window getHandle(common::Window* pValue)
{
    return reinterpret_cast<fg_window>(pValue);
}

fg_font getHandle(common::Font* pValue)
{
    return reinterpret_cast<fg_font>(pValue);
}

fg_image getHandle(common::Image* pValue)
{
    return reinterpret_cast<fg_image>(pValue);
}

fg_chart getHandle(common::Chart* pValue)
{
    return reinterpret_cast<fg_chart>(pValue);
}

fg_histogram getHandle(common::Histogram* pValue)
{
    return reinterpret_cast<fg_histogram>(pValue);
}

fg_plot getHandle(common::Plot* pValue)
{
    return reinterpret_cast<fg_plot>(pValue);
}

fg_surface getHandle(common::Surface* pValue)
{
    return reinterpret_cast<fg_surface>(pValue);
}

common::Window* getWindow(const fg_window& pValue)
{
    return reinterpret_cast<common::Window*>(pValue);
}

common::Font* getFont(const fg_font& pValue)
{
    return reinterpret_cast<common::Font*>(pValue);
}

common::Image* getImage(const fg_image& pValue)
{
    return reinterpret_cast<common::Image*>(pValue);
}

common::Chart* getChart(const fg_chart& pValue)
{
    return reinterpret_cast<common::Chart*>(pValue);
}

common::Histogram* getHistogram(const fg_histogram& pValue)
{
    return reinterpret_cast<common::Histogram*>(pValue);
}

common::Plot* getPlot(const fg_plot& pValue)
{
    return reinterpret_cast<common::Plot*>(pValue);
}

common::Surface* getSurface(const fg_surface& pValue)
{
    return reinterpret_cast<common::Surface*>(pValue);
}
