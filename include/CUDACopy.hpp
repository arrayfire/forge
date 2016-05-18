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

#include <glbinding/gl/gl.h>
using namespace gl;
#ifdef OS_WIN
#include <Windows.h>
#endif
#include <cuda_gl_interop.h>


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

#include <cstdio>

static void handleCUDAError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define FORGE_CUDA_CHECK(err) (handleCUDAError(err, __FILE__, __LINE__ ))

namespace fg
{

template<typename T>
void copy(fg::Image& out, const T * devicePtr)
{
    cudaGraphicsResource *cudaPBOResource;
    FORGE_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPBOResource, out.pbo(), cudaGraphicsMapFlagsWriteDiscard));

    size_t num_bytes;
    T* pboDevicePtr = NULL;

    FORGE_CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPBOResource, 0));
    FORGE_CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&pboDevicePtr, &num_bytes, cudaPBOResource));
    FORGE_CUDA_CHECK(cudaMemcpy(pboDevicePtr, devicePtr, num_bytes, cudaMemcpyDeviceToDevice));
    FORGE_CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));
    FORGE_CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPBOResource));
}

/*
 * Below functions expects OpenGL resource Id and size in bytes to copy the data from
 * CUDA device memory location to graphics memory
 */
template<typename T>
void copy(const int resourceId, const T * devicePtr)
{
    cudaGraphicsResource *cudaVBOResource;
    FORGE_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBOResource, resourceId,
                                                  cudaGraphicsMapFlagsWriteDiscard));
    size_t num_bytes;
    T* vboDevicePtr = NULL;

    FORGE_CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVBOResource, 0));
    FORGE_CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&vboDevicePtr, &num_bytes, cudaVBOResource));
    FORGE_CUDA_CHECK(cudaMemcpy(vboDevicePtr, devicePtr, num_bytes, cudaMemcpyDeviceToDevice));
    FORGE_CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVBOResource, 0));
    FORGE_CUDA_CHECK(cudaGraphicsUnregisterResource(cudaVBOResource));
}

}

#endif

#endif //__CUDA_DATA_COPY_H__
