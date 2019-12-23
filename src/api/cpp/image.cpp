/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/image.h>
#include <fg/window.h>

#include <error.hpp>

#include <utility>

namespace forge {
Image::Image(const unsigned pWidth, const unsigned pHeight,
             const ChannelFormat pFormat, const dtype pDataType)
    : mValue(0) {
    fg_image temp = 0;
    FG_THROW(
        fg_create_image(&temp, pWidth, pHeight, pFormat, (fg_dtype)pDataType));

    std::swap(mValue, temp);
}

Image::Image(const Image& pOther) {
    fg_image temp = 0;

    FG_THROW(fg_retain_image(&temp, pOther.get()));

    std::swap(mValue, temp);
}

Image::Image(const fg_image pHandle) : mValue(pHandle) {}

Image::~Image() { fg_release_image(get()); }

void Image::setAlpha(const float pAlpha) {
    FG_THROW(fg_set_image_alpha(get(), pAlpha));
}

void Image::keepAspectRatio(const bool pKeep) {
    FG_THROW(fg_set_image_aspect_ratio(get(), pKeep));
}

unsigned Image::width() const {
    unsigned temp = 0;
    FG_THROW(fg_get_image_width(&temp, get()));
    return temp;
}

unsigned Image::height() const {
    unsigned temp = 0;
    FG_THROW(fg_get_image_height(&temp, get()));
    return temp;
}

ChannelFormat Image::pixelFormat() const {
    fg_channel_format retVal = (fg_channel_format)0;
    FG_THROW(fg_get_image_pixelformat(&retVal, get()));
    return retVal;
}

forge::dtype Image::channelType() const {
    fg_dtype temp = (fg_dtype)1;
    FG_THROW(fg_get_image_type(&temp, get()));
    return (forge::dtype)temp;
}

unsigned Image::pixels() const {
    unsigned retVal = 0;
    FG_THROW(fg_get_pixel_buffer(&retVal, get()));
    return retVal;
}

unsigned Image::size() const {
    unsigned retVal = 0;
    FG_THROW(fg_get_image_size(&retVal, get()));
    return retVal;
}

void Image::render(const Window& pWindow, const int pX, const int pY,
                   const int pVPW, const int pVPH) const {
    FG_THROW(fg_render_image(pWindow.get(), get(), pX, pY, pVPW, pVPH));
}

fg_image Image::get() const { return mValue; }
}  // namespace forge
