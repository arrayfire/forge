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

class Image {
    private:
        internal::_Image* value;

    public:
        FGAPI Image(unsigned pWidth, unsigned pHeight, ColorMode pFormat, FGType pDataType);
        FGAPI Image(const Image& other);
        FGAPI ~Image();

        FGAPI unsigned width() const;
        FGAPI unsigned height() const;
        FGAPI ColorMode pixelFormat() const;
        FGAPI FGType channelType() const;
        FGAPI unsigned pbo() const;
        FGAPI unsigned size() const;
        FGAPI internal::_Image* get() const;

        FGAPI void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const;
};

}
