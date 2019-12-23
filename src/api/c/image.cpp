/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/defines.hpp>
#include <common/err_handling.hpp>
#include <common/handle.hpp>
#include <common/image.hpp>
#include <common/window.hpp>
#include <fg/image.h>
#include <fg/window.h>

using namespace forge;
using namespace forge::common;

using forge::common::getImage;
using forge::common::getWindow;

fg_err fg_create_image(fg_image* pImage, const unsigned pWidth,
                       const unsigned pHeight, const fg_channel_format pFormat,
                       const fg_dtype pType) {
    try {
        ARG_ASSERT(1, (pWidth > 0));
        ARG_ASSERT(2, (pHeight > 0));

        *pImage = getHandle(
            new common::Image(pWidth, pHeight, pFormat, (forge::dtype)pType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_image(fg_image* pOut, fg_image pImage) {
    try {
        ARG_ASSERT(1, (pImage != 0));

        common::Image* temp = new common::Image(pImage);
        *pOut               = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_image(fg_image pImage) {
    try {
        ARG_ASSERT(0, (pImage != 0));

        delete getImage(pImage);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_image_alpha(fg_image pImage, const float pAlpha) {
    try {
        ARG_ASSERT(0, (pImage != 0));
        ARG_ASSERT(1, (pAlpha >= 0.0 && pAlpha <= 1.0));

        getImage(pImage)->setAlpha(pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_image_aspect_ratio(fg_image pImage, const bool pKeepRatio) {
    try {
        getImage(pImage)->keepAspectRatio(pKeepRatio);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_width(unsigned* pOut, const fg_image pImage) {
    try {
        ARG_ASSERT(1, (pImage != 0));

        *pOut = getImage(pImage)->width();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_height(unsigned* pOut, const fg_image pImage) {
    try {
        ARG_ASSERT(1, (pImage != 0));

        *pOut = getImage(pImage)->height();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_pixelformat(fg_channel_format* pOut,
                                const fg_image pImage) {
    try {
        ARG_ASSERT(1, (pImage != 0));

        *pOut = getImage(pImage)->pixelFormat();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_type(fg_dtype* pOut, const fg_image pImage) {
    try {
        ARG_ASSERT(1, (pImage != 0));

        *pOut = (fg_dtype)(getImage(pImage)->channelType());
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_pixel_buffer(unsigned* pOut, const fg_image pImage) {
    try {
        ARG_ASSERT(1, (pImage != 0));

        *pOut = getImage(pImage)->pbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_image_size(unsigned* pOut, const fg_image pImage) {
    try {
        ARG_ASSERT(1, (pImage != 0));

        *pOut = getImage(pImage)->size();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_render_image(const fg_window pWindow, const fg_image pImage,
                       const int pX, const int pY, const int pWidth,
                       const int pHeight) {
    try {
        ARG_ASSERT(0, (pWindow != 0));
        ARG_ASSERT(1, (pImage != 0));
        ARG_ASSERT(2, (pX >= 0));
        ARG_ASSERT(3, (pY >= 0));
        ARG_ASSERT(4, (pWidth > 0));
        ARG_ASSERT(5, (pHeight > 0));

        getImage(pImage)->render(getWindow(pWindow)->getID(), pX, pY, pWidth,
                                 pHeight, IDENTITY, IDENTITY);
    }
    CATCHALL

    return FG_ERR_NONE;
}
