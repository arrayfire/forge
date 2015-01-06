/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fw/image.h>
#include <image.hpp>
#include <err_common.hpp>

using namespace backend;

fw_err fw_setup_image(ImageHandle *out, const WindowHandle window,
                      const unsigned width, const unsigned height)
{
    try {
        ImageHandle image = new fw_image[1];
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

    return FW_SUCCESS;
}

fw_err fw_draw_image(const ImageHandle in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        drawImage(in);
    }
    CATCHALL;

    return FW_SUCCESS;
}


fw_err fw_destroy_image(const ImageHandle in)
{
    try {
        ARG_ASSERT(0, in != NULL);
        destroyImage(in);
    }
    CATCHALL;

    return FW_SUCCESS;
}

