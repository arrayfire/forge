/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fw/defines.h>

#ifdef __cplusplus
namespace fw
{
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    fw_err fw_create_window(WindowHandle *out, const unsigned height, const unsigned width,
                            const char *title, fw_color_mode mode);

    fw_err fw_destroy_window(const WindowHandle in);
#ifdef __cplusplus
}
#endif
