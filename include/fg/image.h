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
#include <memory>

namespace fg
{

class Image {
    private:
        std::shared_ptr<internal::_Image> value;

    public:
        FGAPI Image(unsigned pWidth, unsigned pHeight, ColorMode pFormat, GLenum pDataType);

        FGAPI unsigned width() const;
        FGAPI unsigned height() const;
        FGAPI ColorMode pixelFormat() const;
        FGAPI GLenum channelType() const;
        FGAPI GLuint pbo() const;
        FGAPI size_t size() const;
        FGAPI internal::_Image* get() const;

        FGAPI void render() const;
};

}
