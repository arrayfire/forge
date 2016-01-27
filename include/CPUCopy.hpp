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

enum BufferType {
    FG_VERTEX_BUFFER = 0,
    FG_COLOR_BUFFER  = 1,
    FG_ALPHA_BUFFER  = 2
};

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
 * `unsigned Renderable::vertices() const;`
 * `unsigned Renderable::verticesSize() const;`
 *
 * Currently fg::Plot, fg::Histogram objects in Forge library fit the bill
 */
template<class Renderable, typename T>
void copy(Renderable& out, const T * dataPtr, const BufferType bufferType=FG_VERTEX_BUFFER)
{
    unsigned rId = 0;
    size_t size = 0;
    switch(bufferType) {
        case FG_VERTEX_BUFFER:
            rId = out.vertices();
            size = out.verticesSize();
            break;
        case FG_COLOR_BUFFER:
            rId = out.colors();
            size = out.colorsSize();
            break;
        case FG_ALPHA_BUFFER:
            rId = out.alphas();
            size = out.alphasSize();
            break;
    }
    glBindBuffer(GL_ARRAY_BUFFER, rId);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, dataPtr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

}

#endif //__CPU_DATA_COPY_H__
