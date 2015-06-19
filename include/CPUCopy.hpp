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

namespace fg
{

template<typename T>
void copy(fg::Image& out, const T * dataPtr)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, out.pbo());
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, out.size(), dataPtr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

/*
 * Below functions takes any renderable forge object that has following member functions
 * defined
 *
 * `unsigned Renderable::vbo() const;`
 * `unsigned Renderable::size() const;`
 *
 * Currently fg::Plot, fg::Histogram objects in Forge library fit the bill
 */
template<class Renderable, typename T>
void copy(Renderable& out, const T * dataPtr)
{
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo());
    glBufferSubData(GL_ARRAY_BUFFER, 0, out.size(), dataPtr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

}

#endif //__CPU_DATA_COPY_H__
