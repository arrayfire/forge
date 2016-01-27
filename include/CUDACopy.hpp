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

#include <cuda_gl_interop.h>
#include <cstdio>

static void handleCUDAError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_ERROR_CHECK(err) (handleCUDAError(err, __FILE__, __LINE__ ))

namespace fg
{

enum BufferType {
    FG_VERTEX_BUFFER = 0,
    FG_COLOR_BUFFER  = 1,
    FG_ALPHA_BUFFER  = 2
};

template<typename T>
void copy(fg::Image& out, const T * devicePtr)
{
    cudaGraphicsResource *cudaPBOResource;
    CUDA_ERROR_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPBOResource, out.pbo(), cudaGraphicsMapFlagsWriteDiscard));

    size_t num_bytes;
    T* pboDevicePtr = NULL;

    CUDA_ERROR_CHECK(cudaGraphicsMapResources(1, &cudaPBOResource, 0));
    CUDA_ERROR_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&pboDevicePtr, &num_bytes, cudaPBOResource));
    CUDA_ERROR_CHECK(cudaMemcpy(pboDevicePtr, devicePtr, num_bytes, cudaMemcpyDeviceToDevice));
    CUDA_ERROR_CHECK(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));
    CUDA_ERROR_CHECK(cudaGraphicsUnregisterResource(cudaPBOResource));
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
void copy(Renderable& out, const T * devicePtr, const BufferType bufferType=FG_VERTEX_BUFFER)
{
    unsigned rId = 0;
    switch(bufferType) {
        case FG_VERTEX_BUFFER:
            rId = out.vertices();
            break;
        case FG_COLOR_BUFFER:
            rId = out.colors();
            break;
        case FG_ALPHA_BUFFER:
            rId = out.alphas();
            break;
    }

    cudaGraphicsResource *cudaVBOResource;
    CUDA_ERROR_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBOResource, rId, cudaGraphicsMapFlagsWriteDiscard));

    size_t num_bytes;
    T* vboDevicePtr = NULL;

    CUDA_ERROR_CHECK(cudaGraphicsMapResources(1, &cudaVBOResource, 0));
    CUDA_ERROR_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&vboDevicePtr, &num_bytes, cudaVBOResource));
    CUDA_ERROR_CHECK(cudaMemcpy(vboDevicePtr, devicePtr, num_bytes, cudaMemcpyDeviceToDevice));
    CUDA_ERROR_CHECK(cudaGraphicsUnmapResources(1, &cudaVBOResource, 0));
    CUDA_ERROR_CHECK(cudaGraphicsUnregisterResource(cudaVBOResource));
}

}

#endif //__CUDA_DATA_COPY_H__
