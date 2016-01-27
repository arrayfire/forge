/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __OPENCL_DATA_COPY_H__
#define __OPENCL_DATA_COPY_H__

namespace fg
{

enum BufferType {
    FG_VERTEX_BUFFER = 0,
    FG_COLOR_BUFFER  = 1,
    FG_ALPHA_BUFFER  = 2
};

static void copy(fg::Image& out, const cl::Buffer& in, const cl::CommandQueue& queue)
{
    cl::BufferGL pboMapBuffer(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_WRITE_ONLY, out.pbo(), NULL);

    std::vector<cl::Memory> shared_objects;
    shared_objects.push_back(pboMapBuffer);

    glFinish();
    queue.enqueueAcquireGLObjects(&shared_objects);
    queue.enqueueCopyBuffer(in, pboMapBuffer, 0, 0, out.size(), NULL, NULL);
    queue.finish();
    queue.enqueueReleaseGLObjects(&shared_objects);
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
template<class Renderable>
void copy(Renderable& out, const cl::Buffer& in, const cl::CommandQueue& queue,
          const BufferType bufferType=FG_VERTEX_BUFFER)
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

    cl::BufferGL vboMapBuffer(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_WRITE_ONLY, rId, NULL);

    std::vector<cl::Memory> shared_objects;
    shared_objects.push_back(vboMapBuffer);

    glFinish();
    queue.enqueueAcquireGLObjects(&shared_objects);
    queue.enqueueCopyBuffer(in, vboMapBuffer, 0, 0, size, NULL, NULL);
    queue.finish();
    queue.enqueueReleaseGLObjects(&shared_objects);
}

}

#endif //__OPENCL_DATA_COPY_H__
