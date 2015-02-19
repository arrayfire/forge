/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <afgfx/image.h>
#include <image.hpp>
#include <err_common.hpp>

using namespace backend;

afgfx_err afgfx_setup_image(afgfx_image *out, const afgfx_window window,
                      const unsigned width, const unsigned height)
{
    try {
        afgfx_image image = new afgfx_image_struct[1];
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

    return AFGFX_SUCCESS;
}

afgfx_err afgfx_draw_image(const afgfx_image in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        drawImage(in);
    }
    CATCHALL;

    return AFGFX_SUCCESS;
}


afgfx_err afgfx_destroy_image(const afgfx_image in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        destroyImage(in);
    }
    CATCHALL;

    return AFGFX_SUCCESS;
}

