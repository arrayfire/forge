/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/window.h>
#include <window.hpp>
#include <err_common.hpp>

using namespace backend;

fg_err fg_create_window(fg_window_handle *out, const unsigned width, const unsigned height, const char *title,
                              fg_color_mode mode, GLenum type)
{
    try {
        fg_window_handle window;
        DIM_ASSERT(1, height > 0);
        DIM_ASSERT(2, width > 0);

        switch(type) {
            case GL_FLOAT:          window = createWindow<float>(width, height, title, mode);  break;
            case GL_INT:            window = createWindow<int  >(width, height, title, mode);  break;
            case GL_UNSIGNED_INT:   window = createWindow<uint >(width, height, title, mode);  break;
            case GL_BYTE:           window = createWindow<char >(width, height, title, mode);  break;
            case GL_UNSIGNED_BYTE:  window = createWindow<uchar>(width, height, title, mode);  break;
            default:  TYPE_ERROR(1, type);
        }
        window->type = type;
        std::swap(*out, window);
    }
    CATCHALL;

    return FG_SUCCESS;
}

fg_err fg_make_window_current(const fg_window_handle in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        makeWindowCurrent(in);
    }
    CATCHALL;

    return FG_SUCCESS;
}
fg_err fg_destroy_window(const fg_window_handle in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        destroyWindow(in);
    }
    CATCHALL;

    return FG_SUCCESS;
}
