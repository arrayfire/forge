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


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

namespace fg
{

static
void copy(fg::Image& out, const cl::Buffer& in, const cl::CommandQueue& queue)
{
    cl::Event ev;
    cl::BufferGL pboMapBuffer(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_WRITE_ONLY, out.pbo(), NULL);

    std::vector<cl::Memory> shared_objects;
    shared_objects.push_back(pboMapBuffer);

    glFinish();
    cl_int res = queue.enqueueAcquireGLObjects(&shared_objects, NULL, &ev);
    ev.wait();

    queue.enqueueCopyBuffer(in, pboMapBuffer, 0, 0, out.size(), NULL, NULL);
    res = queue.enqueueReleaseGLObjects(&shared_objects, NULL, &ev);
    queue.finish();
}

/*
 * Below functions expects OpenGL resource Id and size in bytes to copy the data from
 * OpenCL Buffer to graphics memory
 */
static
void copy(const int resourceId, const ::size_t resourceSize,
          const cl::Buffer& in, const cl::CommandQueue& queue)
{
    cl::Event ev;
    cl::BufferGL vboMapBuffer(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_WRITE_ONLY, resourceId, NULL);

    std::vector<cl::Memory> shared_objects;
    shared_objects.push_back(vboMapBuffer);

    glFinish();
    cl_int res = queue.enqueueAcquireGLObjects(&shared_objects, NULL, &ev);
    ev.wait();

    queue.enqueueCopyBuffer(in, vboMapBuffer, 0, 0, resourceSize, NULL, NULL);
    res = queue.enqueueReleaseGLObjects(&shared_objects, NULL, &ev);
    queue.finish();
}

}

#pragma GCC diagnostic pop

#endif

#endif //__OPENCL_DATA_COPY_H__
