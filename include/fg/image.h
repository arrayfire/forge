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

/**
   \class Image
 */
class Image {
    private:
        internal::_Image* value;

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
        FGAPI Image(unsigned pWidth, unsigned pHeight, ChannelFormat pFormat, dtype pDataType);

        /**
           Copy constructor of Image

           \param[in] other is the Image of which we make a copy of.
         */
        FGAPI Image(const Image& other);

        /**
           Image Destructor
         */
        FGAPI ~Image();

        /**
           Get Image width
           \return image width
         */
        FGAPI unsigned width() const;

        /**
           Get Image height
           \return image width
         */
        FGAPI unsigned height() const;

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
        FGAPI unsigned pbo() const;

        /**
           Get the OpenGL Pixel Buffer Object resource size

           \return OpenGL PBO resource size.
         */
        FGAPI unsigned size() const;

        /**
           Get the handle to internal implementation of Image
         */
        FGAPI internal::_Image* get() const;
};

}
