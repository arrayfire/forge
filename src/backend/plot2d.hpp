/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot2d.h>

namespace backend
{
    fg_plot_handle plot_init(const fg_window_handle window, const unsigned width, const unsigned height);

    void plot_2d(fg_plot_handle plot, const double xmax, const double xmin, const double ymax, const double ymin);

    void destroyPlot(fg_plot_handle plot);
}
