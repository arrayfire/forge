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


#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup image_functions
 *  @{
 */

/**
   Create a Image object

   \param[out] pImage will be set to created Image object
   \param[in] pWidth Width of the image
   \param[in] pHeight Height of the image
   \param[in] pFormat Color channel format of image, uses one of the values
              of \ref fg_channel_format
   \param[in] pType takes one of the values of \ref fg_dtype that indicates
              the integral data type of histogram data

   \return \ref fg_err error code
 */
FGAPI fg_err fg_create_image(fg_image* pImage,
                             const unsigned pWidth, const unsigned pHeight,
                             const fg_channel_format pFormat, const fg_dtype pType);

/**
   Destroy image object

   \param[in] pImage is the image handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_destroy_image(fg_image pImage);

/**
   Set a global alpha value for rendering the image

   \param[in] pImage is the image handle
   \param[in] pAlpha

   \return \ref fg_err error code
 */
FGAPI fg_err fg_set_image_alpha(fg_image pImage, const float pAlpha);

/**
   Set option to inform whether to maintain aspect ratio of original image

   \param[in] pImage is the image handle
   \param[in] pKeepRatio informs the image object if original aspect ratio has to be maintained

   \return \ref fg_err error code
 */
FGAPI fg_err fg_set_image_aspect_ratio(fg_image pImage, const bool pKeepRatio);

/**
   Get the width of the image

   \param[out] pOut will be set to the width of the image
   \param[in] pImage is the image handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_image_width(unsigned *pOut, const fg_image pImage);

/**
   Get the height of the image

   \param[out] pOut will be set to the height of the image
   \param[in] pImage is the image handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_image_height(unsigned *pOut, const fg_image pImage);

/**
   Get the channel format of the image

   \param[out] pOut will be set to the channel format of the image
   \param[in] pImage is the image handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_image_pixelformat(fg_channel_format* pOut, const fg_image pImage);

/**
   Get the pixel data type of the image

   \param[out] pOut will be set to the pixel data type of the image
   \param[in] pImage is the image handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_image_type(fg_dtype* pOut, const fg_image pImage);

/**
   Get the image buffer resource identifier

   \param[out] pOut will be set to the image resource identifier
   \param[in] pImage is the image handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_pixel_buffer(unsigned* pOut, const fg_image pImage);

/**
   Get the image buffer size in bytes

   \param[out] pOut will be set to the image buffer size in bytes
   \param[in] pImage is the image handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_get_image_size(unsigned* pOut, const fg_image pImage);

/**
   Render the image to given window

   \param[in] pWindow is target window to where image will be rendered
   \param[in] pImage is the image handle
   \param[in] pX is x coordinate of origin of viewport in window coordinates
   \param[in] pY is y coordinate of origin of viewport in window coordinates
   \param[in] pWidth is the width of the viewport
   \param[in] pHeight is the height of the viewport

   \return \ref fg_err error code
 */
FGAPI fg_err fg_render_image(const fg_window pWindow,
                             const fg_image pImage,
                             const int pX, const int pY, const int pWidth, const int pHeight);

/** @} */

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

namespace forge
{

class Window;

/**
   \class Image

   \brief Image is plain rendering of an image over the window or sub-region of it.
 */
class Image {
    private:
        fg_image mValue;

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
        FGAPI Image(const unsigned pWidth, const unsigned pHeight,
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
           Get the resource id of image buffer

           \return image buffer id
         */
        FGAPI unsigned pixels() const;

        /**
           Get the image data size in bytes

           \return image buffer size in bytes
         */
        FGAPI unsigned size() const;

        /**
           Render the image to given window

           \param[in] pWindow is target window to where image will be rendered
           \param[in] pX is x coordinate of origin of viewport in window coordinates
           \param[in] pY is y coordinate of origin of viewport in window coordinates
           \param[in] pVPW is the width of the viewport
           \param[in] pVPH is the height of the viewport
         */
        FGAPI void render(const Window& pWindow,
                          const int pX, const int pY, const int pVPW, const int pVPH) const;

        /**
           Get the handle to internal implementation of Image
         */
        FGAPI fg_image get() const;
};

}

#endif
