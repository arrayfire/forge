/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/image.h>
#include "error.hpp"

namespace fg
{

Image::Image() :mHandle(0) {}

Image::Image(const uint pWidth, const uint pHeight, const Window& pWindow)
{
    FG_THROW(fg_setup_image(&mHandle, pWindow.get(), pWidth, pHeight));
}

Image::~Image()
{
    FG_THROW(fg_destroy_image(mHandle));
}

uint Image::width() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Image Handle");
    return mHandle->src_width;
}

uint Image::height() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Image Handle");
    return mHandle->src_height;
}

FGuint Image::pboResourceId() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Image Handle");
    return mHandle->gl_PBO;
}

FGuint Image::texResourceId() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Image Handle");
    return mHandle->gl_Tex;
}

FGuint Image::shaderResourceId() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Image Handle");
    return mHandle->gl_Shader;
}

FGenum Image::pixelFormat() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Image Handle");
    return mHandle->gl_Format;
}

FGenum Image::channelType() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Image Handle");
    return mHandle->gl_Type;
}

fg_image_handle Image::get() const
{
    return mHandle;
}

void drawImage(const Image& pImage)
{
    FG_THROW(fg_draw_image(pImage.get()));
}

}
