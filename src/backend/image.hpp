/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fw/image.h>

namespace backend
{
    template<typename T>
    ImageHandle setupImage(WindowHandle window, const unsigned height, const unsigned width);

    void drawImage(const ImageHandle image);

    void destroyImage(const ImageHandle image);
}
