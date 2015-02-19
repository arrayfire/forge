/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <afgfx/window.h>
#include <window.hpp>
#include <err_common.hpp>

using namespace backend;

afgfx_err afgfx_create_window(afgfx_window *out, const unsigned width, const unsigned height, const char *title,
                              afgfx_color_mode mode, GLenum type)
{
    try {
        afgfx_window window;
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

    return AFGFX_SUCCESS;
}

afgfx_err afgfx_make_window_current(const afgfx_window in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        makeWindowCurrent(in);
    }
    CATCHALL;

    return AFGFX_SUCCESS;
}
afgfx_err afgfx_destroy_window(const afgfx_window in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        destroyWindow(in);
    }
    CATCHALL;

    return AFGFX_SUCCESS;
}
