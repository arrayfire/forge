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

static
void copy(fg::Image& out, const cl::Buffer& in, const cl::CommandQueue& queue)
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
 * Below functions expects OpenGL resource Id and size in bytes to copy the data from
 * OpenCL Buffer to graphics memory
 */
static
void copy(const int resourceId, const size_t resourceSize,
          const cl::Buffer& in, const cl::CommandQueue& queue)
{
    cl::BufferGL vboMapBuffer(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_WRITE_ONLY, resourceId, NULL);

    std::vector<cl::Memory> shared_objects;
    shared_objects.push_back(vboMapBuffer);

    glFinish();
    queue.enqueueAcquireGLObjects(&shared_objects);
    queue.enqueueCopyBuffer(in, vboMapBuffer, 0, 0, resourceSize, NULL, NULL);
    queue.finish();
    queue.enqueueReleaseGLObjects(&shared_objects);
}

}

#endif //__OPENCL_DATA_COPY_H__
