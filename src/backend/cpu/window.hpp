/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fw/defines.h>

namespace cpu
{
    template<typename T>
    WindowHandle createWindow(const unsigned height, const unsigned width, const char *title,
                              fw_color_mode mode);

    void destroyWindow(const WindowHandle window);
}
