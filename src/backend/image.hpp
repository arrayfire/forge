/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/image.h>

namespace backend
{
    template<typename T>
    fg_image_handle setupImage(fg_window_handle window, const unsigned width, const unsigned height);

    void drawImage(const fg_image_handle image);

    void destroyImage(const fg_image_handle image);
}
