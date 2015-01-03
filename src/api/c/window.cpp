/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fw/window.h>
#include <window.hpp>
#include <err_common.hpp>

using namespace backend;

fw_err fw_create_window(WindowHandle *out, const unsigned height, const unsigned width, const char *title,
                        fw_color_mode mode, fw_dtype type)
{
    try {
        WindowHandle window;
        DIM_ASSERT(1, height > 0);
        DIM_ASSERT(2, width > 0);

        switch(type) {
            case f32: window = createWindow<float  >(height, width, title, mode);  break;
            case f64: window = createWindow<double >(height, width, title, mode);  break;
            case b8:  window = createWindow<char   >(height, width, title, mode);  break;
            case s32: window = createWindow<int    >(height, width, title, mode);  break;
            case u32: window = createWindow<uint   >(height, width, title, mode);  break;
            case u8:  window = createWindow<uchar  >(height, width, title, mode);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out, window);
    }
    CATCHALL;

    return FW_SUCCESS;
}

fw_err fw_destroy_window(const WindowHandle in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        destroyWindow(in);
    }
    CATCHALL;

    return FW_SUCCESS;
}
