/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __CPU_DATA_COPY_H__
#define __CPU_DATA_COPY_H__


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

#include <glbinding/gl/gl.h>

namespace fg
{

static
void copy(fg::Image& out, const void * dataPtr)
{
    gl::glBindBuffer(gl::GL_PIXEL_UNPACK_BUFFER, out.pbo());
    gl::glBufferSubData(gl::GL_PIXEL_UNPACK_BUFFER, 0, out.size(), dataPtr);
    gl::glBindBuffer(gl::GL_PIXEL_UNPACK_BUFFER, 0);
}

/*
 * Below functions expects OpenGL resource Id and size in bytes to copy the data from
 * cpu memory location to graphics memory
 */
static
void copy(const int resourceId, const size_t resourceSize, const void * dataPtr)
{
    gl::glBindBuffer(gl::GL_ARRAY_BUFFER, resourceId);
    gl::glBufferSubData(gl::GL_ARRAY_BUFFER, 0, resourceSize, dataPtr);
    gl::glBindBuffer(gl::GL_ARRAY_BUFFER, 0);
}

}

#endif

#endif //__CPU_DATA_COPY_H__
