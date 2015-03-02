/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/window.h>

namespace backend
{
    template<typename T>
    fg_window_handle createWindow(const unsigned width, const unsigned height, const char *title,
                              fg_color_mode mode);

    void makeWindowCurrent(const fg_window_handle window);

    void destroyWindow(const fg_window_handle window);
}
