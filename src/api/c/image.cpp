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
#include <err_common.hpp>
#include <image.hpp>
#include <window.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace forge;

fg_err fg_create_image(fg_image* pImage,
                       const unsigned pWidth, const unsigned pHeight,
                       const fg_channel_format pFormat, const fg_dtype pType)
{
    try {
        *pImage = getHandle(new common::Image(pWidth, pHeight, pFormat, (forge::dtype)pType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_destroy_image(fg_image pImage)
{
    try {
        delete getImage(pImage);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_image_alpha(fg_image pImage, const float pAlpha)
{
    try {
        getImage(pImage)->setAlpha(pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_image_aspect_ratio(fg_image pImage, const bool pKeepRatio)
{
    try {
        getImage(pImage)->keepAspectRatio(pKeepRatio);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_width(unsigned *pOut, const fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->width();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_height(unsigned *pOut, const fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->height();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_pixelformat(fg_channel_format* pOut, const fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->pixelFormat();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_type(fg_dtype* pOut, const fg_image pImage)
{
    try {
        *pOut = (fg_dtype)(getImage(pImage)->channelType());
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_pbo(unsigned* pOut, const fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->pbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_pbo_size(unsigned* pOut, const fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->size();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_render_image(const fg_window pWindow,
                       const fg_image pImage,
                       const int pX, const int pY, const int pWidth, const int pHeight)
{
    try {
        getImage(pImage)->render(getWindow(pWindow)->getID(),
                                 pX, pY, pWidth, pHeight,
                                 IDENTITY, IDENTITY);
    }
    CATCHALL

    return FG_ERR_NONE;
}
