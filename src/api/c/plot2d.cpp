/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <fg/plot2d.h>
#include <plot2d.hpp>
#include <err_common.hpp>

using namespace backend;

fg_err fg_plot_init(fg_plot_handle *in, const fg_window_handle window, const unsigned width, const unsigned height)
{
    fg_plot_handle plot = new fg_plot_struct[1];
    plot = plot_init(window, width, height);
    std::swap(*in, plot);
    return FG_SUCCESS;
}

fg_err fg_plot2d(fg_plot_handle plot, const double xmax, const double xmin, const double ymax, const double ymin, const int size)
{
    plot_2d(plot, xmax, xmin,ymax, ymin, size);
    return FG_SUCCESS;
}

fg_err fg_destroy_plot(fg_plot_handle plot)
{
    destroyPlot(plot);
    return FG_SUCCESS;
}

