/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <fg/defines.h>

namespace internal
{
class _Image;
}

namespace fg
{

class Window;

/**
   \class Image

   \brief Image is plain rendering of an image over the window or sub-region of it.
 */
class Image {
    private:
        internal::_Image* mValue;

    public:
        /**
           Creates a Image object

           \param[in] pWidth Width of the image
           \param[in] pHeight Height of the image
           \param[in] pFormat Color channel format of image, uses one of the values
                      of \ref ChannelFormat
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of histogram data
         */
        FGAPI Image(const uint pWidth, const uint pHeight,
                    const ChannelFormat pFormat=FG_RGBA, const dtype pDataType=f32);

        /**
           Copy constructor of Image

           \param[in] pOther is the Image of which we make a copy of.
         */
        FGAPI Image(const Image& pOther);

        /**
           Image Destructor
         */
        FGAPI ~Image();

        /**
           Set a global alpha value for rendering the image

           \param[in] pAlpha
         */
        FGAPI void setAlpha(const float pAlpha);

        /**
           Set option to inform whether to maintain aspect ratio of original image

           \param[in] pKeep
         */
        FGAPI void keepAspectRatio(const bool pKeep);

        /**
           Get Image width
           \return image width
         */
        FGAPI uint width() const;

        /**
           Get Image height
           \return image width
         */
        FGAPI uint height() const;

        /**
           Get Image's channel format
           \return \ref ChannelFormat value of Image
         */
        FGAPI ChannelFormat pixelFormat() const;

        /**
           Get Image's integral data type
           \return \ref dtype value of Image
         */
        FGAPI dtype channelType() const;

        /**
           Get the OpenGL Pixel Buffer Object identifier

           \return OpenGL PBO resource id.
         */
        FGAPI uint pbo() const;

        /**
           Get the OpenGL Pixel Buffer Object resource size

           \return OpenGL PBO resource size.
         */
        FGAPI uint size() const;

        /**
           Render the image to given window

           \param[in] pWindow is target window to where image will be rendered
           \param[in] pX is x coordinate of origin of viewport in window coordinates
           \param[in] pY is y coordinate of origin of viewport in window coordinates
           \param[in] pVPW is the width of the viewport
           \param[in] pVPH is the height of the viewport
           \param[in] pTransform is an array of floats. This array is expected to contain
                      at least 16 elements

           Note: pTransform array is assumed to be of expected length.
         */
        FGAPI void render(const Window& pWindow,
                          const int pX, const int pY, const int pVPW, const int pVPH,
                          const float* pTransform) const;

        /**
           Get the handle to internal implementation of Image
         */
        FGAPI internal::_Image* get() const;
};

}
