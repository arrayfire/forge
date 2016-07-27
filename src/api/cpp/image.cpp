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

#include <handle.hpp>
#include <image.hpp>
#include <window.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace fg
{

Image::Image(const uint pWidth, const uint pHeight,
             const ChannelFormat pFormat, const dtype pDataType)
{
    mValue = getHandle(new common::Image(pWidth, pHeight, pFormat, pDataType));
}

Image::Image(const Image& pOther)
{
    mValue = getHandle(new common::Image(pOther.get()));
}

Image::~Image()
{
    delete getImage(mValue);
}

void Image::setAlpha(const float pAlpha)
{
    getImage(mValue)->setAlpha(pAlpha);
}

void Image::keepAspectRatio(const bool pKeep)
{
    getImage(mValue)->keepAspectRatio(pKeep);
}

uint Image::width() const
{
    return getImage(mValue)->width();
}

uint Image::height() const
{
    return getImage(mValue)->height();
}

ChannelFormat Image::pixelFormat() const
{
    return getImage(mValue)->pixelFormat();
}

fg::dtype Image::channelType() const
{
    return getImage(mValue)->channelType();
}

uint Image::pbo() const
{
    return getImage(mValue)->pbo();
}

uint Image::size() const
{
    return (uint)getImage(mValue)->size();
}

void Image::render(const Window& pWindow,
                   const int pX, const int pY, const int pVPW, const int pVPH) const
{
    getImage(mValue)->render(getWindow(pWindow.get())->getID(),
                             pX, pY, pVPW, pVPH,
                             IDENTITY, IDENTITY);
}


fg_image Image::get() const
{
    return mValue;
}

}
