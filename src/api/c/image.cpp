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
#include <Image.hpp>
#include <Window.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

fg_err fg_create_image(fg_image* pImage,
                       const uint pWidth, const uint pHeight,
                       const fg_channel_format pFormat, const fg_dtype pType)
{
    try {
        *pImage = getHandle(new common::Image(pWidth, pHeight, pFormat, pType));
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_destroy_image(fg_image pImage)
{
    try {
        delete getImage(pImage);
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_set_image_alpha(fg_image pImage, const float pAlpha)
{
    try {
        getImage(pImage)->setAlpha(pAlpha);
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_set_image_aspect_ratio(fg_image pImage, const bool pKeepRatio)
{
    try {
        getImage(pImage)->keepAspectRatio(pKeepRatio);
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_get_image_width(uint *pOut, fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->width();
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_get_image_height(uint *pOut, fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->height();
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_get_image_pixelformat(fg_channel_format* pOut, fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->pixelFormat();
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_get_image_type(fg_dtype* pOut, fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->channelType();
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_get_image_pbo(uint* pOut, fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->pbo();
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_get_image_pbo_size(uint* pOut, fg_image pImage)
{
    try {
        *pOut = getImage(pImage)->size();
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_render_image(fg_window pWindow,
                       const fg_image pImage,
                       const int pX, const int pY, const int pWidth, const int pHeight,
                       const float* pTransform)
{
    try {
        getImage(pImage)->render(getWindow(pWindow)->getID(),
                                 pX, pY, pWidth, pHeight,
                                 glm::make_mat4(pTransform));
    }
    CATCHALL

    return FG_SUCCESS;
}
