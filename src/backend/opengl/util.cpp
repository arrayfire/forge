/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/util.h>
#include <common.hpp>
#include <err_opengl.hpp>

using namespace gl;

fg_err fg_update_vertex_buffer(const unsigned pBufferId,
                               const size_t pBufferSize,
                               const void* pBufferData)
{
    try {
        glBindBuffer(GL_ARRAY_BUFFER, pBufferId);
        glBufferSubData(GL_ARRAY_BUFFER, 0, pBufferSize, pBufferData);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_update_pixel_buffer(const unsigned pBufferId,
                              const size_t pBufferSize,
                              const void* pBufferData)
{
    try {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pBufferId);
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, pBufferSize, pBufferData);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_finish()
{
    try {
        glFinish();
    }
    CATCHALL

    return FG_ERR_NONE;
}

namespace fg
{

void updateVertexBuffer(const unsigned pBufferId,
                        const size_t pBufferSize,
                        const void* pBufferData)
{
    fg_err val = fg_update_vertex_buffer(pBufferId, pBufferSize, pBufferData);
    if (val!=FG_ERR_NONE)
        throw fg::Error(__PRETTY_FUNCTION__, __LINE__, "Vertex Buffer Object update failed", val);
}

void updatePixelBuffer(const unsigned pBufferId,
                       const size_t pBufferSize,
                       const void* pBufferData)
{
    fg_err val = fg_update_pixel_buffer(pBufferId, pBufferSize, pBufferData);
    if (val!=FG_ERR_NONE)
        throw fg::Error(__PRETTY_FUNCTION__, __LINE__, "Pixel Buffer Object update failed", val);
}

void finish()
{
    fg_err val = fg_finish();
    if (val!=FG_ERR_NONE)
        throw fg::Error(__PRETTY_FUNCTION__, __LINE__, "glFinish failed", val);
}

}
