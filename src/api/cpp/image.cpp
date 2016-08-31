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

namespace forge
{

Image::Image(const unsigned pWidth, const unsigned pHeight,
             const ChannelFormat pFormat, const dtype pDataType)
{
    try {
        mValue = getHandle(new common::Image(pWidth, pHeight, pFormat, pDataType));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Image::Image(const Image& pOther)
{
    try {
        mValue = getHandle(new common::Image(pOther.get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Image::~Image()
{
    delete getImage(mValue);
}

void Image::setAlpha(const float pAlpha)
{
    try {
        getImage(mValue)->setAlpha(pAlpha);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Image::keepAspectRatio(const bool pKeep)
{
    try {
        getImage(mValue)->keepAspectRatio(pKeep);
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Image::width() const
{
    try {
        return getImage(mValue)->width();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Image::height() const
{
    try {
        return getImage(mValue)->height();
    } CATCH_INTERNAL_TO_EXTERNAL
}

ChannelFormat Image::pixelFormat() const
{
    try {
        return getImage(mValue)->pixelFormat();
    } CATCH_INTERNAL_TO_EXTERNAL
}

forge::dtype Image::channelType() const
{
    try {
        return getImage(mValue)->channelType();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Image::pbo() const
{
    try {
        return getImage(mValue)->pbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned Image::size() const
{
    try {
        return (unsigned)getImage(mValue)->size();
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Image::render(const Window& pWindow,
                   const int pX, const int pY, const int pVPW, const int pVPH) const
{
    try {
        getImage(mValue)->render(getWindow(pWindow.get())->getID(),
                                 pX, pY, pVPW, pVPH,
                                 IDENTITY, IDENTITY);
    } CATCH_INTERNAL_TO_EXTERNAL
}


fg_image Image::get() const
{
    try {
        return mValue;
    } CATCH_INTERNAL_TO_EXTERNAL
}

}
