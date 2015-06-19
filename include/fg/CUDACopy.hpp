/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __CUDA_DATA_COPY_H__
#define __CUDA_DATA_COPY_H__

namespace fg
{

template<typename T>
void copy(fg::Image& out, const T * devicePtr)
{
    cudaGraphicsResource *cudaPBOResource;
    cudaGraphicsGLRegisterBuffer(&cudaPBOResource, out.pbo(), cudaGraphicsMapFlagsWriteDiscard);

    size_t num_bytes;
    T* pboDevicePtr = NULL;

    cudaGraphicsMapResources(1, &cudaPBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&pboDevicePtr, &num_bytes, cudaPBOResource);
    cudaMemcpy(pboDevicePtr, devicePtr, num_bytes, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);
    cudaGraphicsUnregisterResource(cudaPBOResource);
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
void copy(Renderable& out, const T * devicePtr)
{
    cudaGraphicsResource *cudaVBOResource;
    cudaGraphicsGLRegisterBuffer(&cudaVBOResource, out.vbo(), cudaGraphicsMapFlagsWriteDiscard);

    size_t num_bytes;
    T* vboDevicePtr = NULL;

    cudaGraphicsMapResources(1, &cudaVBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&vboDevicePtr, &num_bytes, cudaVBOResource);
    cudaMemcpy(vboDevicePtr, devicePtr, num_bytes, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
    cudaGraphicsUnregisterResource(cudaVBOResource);
}

}

#endif //__CUDA_DATA_COPY_H__
