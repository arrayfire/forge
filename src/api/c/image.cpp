/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/image.h>
#include <image.hpp>
#include <err_common.hpp>

using namespace backend;

fg_err fg_setup_image(fg_image_handle *out, const fg_window_handle window,
                      const unsigned width, const unsigned height)
{
    try {
        fg_image_handle image = new fg_image_struct[1];
        switch(window->type) {
            case GL_FLOAT:           image = setupImage<float>(window, width, height);  break;
            case GL_INT:             image = setupImage<int  >(window, width, height);  break;
            case GL_UNSIGNED_INT:    image = setupImage<uint >(window, width, height);  break;
            case GL_BYTE:            image = setupImage<char >(window, width, height);  break;
            case GL_UNSIGNED_BYTE:   image = setupImage<uchar>(window, width, height);  break;
            default:  TYPE_ERROR(1, window->type);
        }
        std::swap(*out, image);
    }
    CATCHALL;

    return FG_SUCCESS;
}

fg_err fg_draw_image(const fg_image_handle in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        drawImage(in);
    }
    CATCHALL;

    return FG_SUCCESS;
}


fg_err fg_destroy_image(const fg_image_handle in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        destroyImage(in);
    }
    CATCHALL;

    return FG_SUCCESS;
}

