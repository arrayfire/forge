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

FGAPI fg_err fg_update_vertex_buffer(const unsigned pBufferId,
                                     const size_t pBufferSize,
                                     const void* pBufferData);

FGAPI fg_err fg_update_pixel_buffer(const unsigned pBufferId,
                                    const size_t pBufferSize,
                                    const void* pBufferData);

FGAPI fg_err fg_finish();

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace fg
{

FGAPI void updateVertexBuffer(const unsigned pBufferId,
                              const size_t pBufferSize,
                              const void* pBufferData);

FGAPI void updatePixelBuffer(const unsigned pBufferId,
                             const size_t pBufferSize,
                             const void* pBufferData);

FGAPI void finish();

}
#endif
