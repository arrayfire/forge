/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __COMPUTE_DATA_COPY_H__
#define __COMPUTE_DATA_COPY_H__

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


#if defined(USE_FORGE_CPU_COPY_HELPERS)

// No special headers for cpu backend

#elif defined(USE_FORGE_CUDA_COPY_HELPERS)

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#elif defined(USE_FORGE_OPENCL_COPY_HELPERS)

// No special headers for opencl backend

#else

    #error "Invalid Compute model, exiting."

#endif


/** A backend-agnostic handle to a compute memory resource originating from an OpenGL resource.

    - cudaGraphicsResource in CUDA
    - cl_mem in OpenCL
    - GLuint from standard cpu
  */
#if defined(USE_FORGE_CPU_COPY_HELPERS)
typedef GLuint GfxResourceHandle;
#elif defined(USE_FORGE_CUDA_COPY_HELPERS)
typedef cudaGraphicsResource* GfxResourceHandle;
#elif defined(USE_FORGE_OPENCL_COPY_HELPERS)
typedef cl_mem GfxResourceHandle;
#endif


/** A backend-agnostic handle to a compute memory resource.

  For example:
    CUDA device pointer, like float*, int* from cudaMalloc.
    A cl_mem* from OpenCL's clCreateBuffer
  */
typedef void* ComputeResourceHandle;

typedef enum {
    FORGE_PBO = 0,     ///< OpenGL Pixel Buffer Object
    FORGE_VBO = 1      ///< OpenGL Vertex Buffer Object
} BufferType;

typedef struct {
    GfxResourceHandle mId;
    BufferType mTarget;
} GfxHandle;


///////////////////////////////////////////////////////////////////////////////

#if defined(USE_FORGE_CPU_COPY_HELPERS)

static
void createGLBuffer(GfxHandle** pOut, const unsigned pResourceId, const BufferType pTarget)
{
    GfxHandle* temp = (GfxHandle*)malloc(sizeof(GfxHandle));

    temp->mId = pResourceId;
    temp->mTarget = pTarget;

    *pOut = temp;
}

static
void releaseGLBuffer(GfxHandle* pHandle)
{
    free(pHandle);
}

static
void copyToGLBuffer(GfxHandle* pGLDestination, ComputeResourceHandle  pSource, const size_t pSize)
{
    GfxHandle* temp = pGLDestination;

    GLenum target = (temp->mTarget==FORGE_PBO ? GL_PIXEL_UNPACK_BUFFER : GL_ARRAY_BUFFER);

    glBindBuffer(target, temp->mId);
    glBufferSubData(target, 0, pSize, pSource);
    glBindBuffer(target, 0);
}
#endif

///////////////////////////////////////////////////////////////////////////////

#if defined(USE_FORGE_CUDA_COPY_HELPERS)

static void handleCUDAError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define FORGE_CUDA_CHECK(err) (handleCUDAError(err, __FILE__, __LINE__ ))

static
void createGLBuffer(GfxHandle** pOut, const unsigned pResourceId, const BufferType pTarget)
{
    GfxHandle* temp = (GfxHandle*)malloc(sizeof(GfxHandle));

    temp->mTarget = pTarget;

    cudaGraphicsResource *cudaPBOResource;

    FORGE_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPBOResource,
                                                  pResourceId,
                                                  cudaGraphicsMapFlagsWriteDiscard));

    temp->mId = cudaPBOResource;

    *pOut = temp;
}

static
void releaseGLBuffer(GfxHandle* pHandle)
{
    FORGE_CUDA_CHECK(cudaGraphicsUnregisterResource(pHandle->mId));
    free(pHandle);
}

static
void copyToGLBuffer(GfxHandle* pGLDestination, ComputeResourceHandle  pSource, const size_t pSize)
{
    size_t numBytes;
    void* pointer = NULL;

    cudaGraphicsResource *cudaResource = pGLDestination->mId;

    FORGE_CUDA_CHECK(cudaGraphicsMapResources(1, &cudaResource, 0));

    FORGE_CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&pointer, &numBytes, cudaResource));

    FORGE_CUDA_CHECK(cudaMemcpy(pointer, pSource, numBytes, cudaMemcpyDeviceToDevice));

    FORGE_CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
}
#endif

///////////////////////////////////////////////////////////////////////////////

#if defined(USE_FORGE_OPENCL_COPY_HELPERS)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#define FORGE_OCL_CHECK(cl_status, message) \
    if(cl_status != CL_SUCCESS) \
    { \
        printf("Error: %s \nError Code: %d\n", message, cl_status);\
        printf("Location: %s:%i\n", __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                             \
    }

static
void createGLBuffer(GfxHandle** pOut, const unsigned pResourceId, const BufferType pTarget)
{
    GfxHandle* temp = (GfxHandle*)malloc(sizeof(GfxHandle));

    temp->mTarget = pTarget;

    cl_int returnCode = CL_SUCCESS;

    temp->mId = clCreateFromGLBuffer(getContext(), CL_MEM_WRITE_ONLY, pResourceId, &returnCode);

    FORGE_OCL_CHECK(returnCode, "Failed in clCreateFromGLBuffer");

    *pOut = temp;
}

static
void releaseGLBuffer(GfxHandle* pHandle)
{
    FORGE_OCL_CHECK(clReleaseMemObject(pHandle->mId), "Failed in clReleaseMemObject");
    free(pHandle);
}

static
void copyToGLBuffer(GfxHandle* pGLDestination, ComputeResourceHandle  pSource, const size_t pSize)
{
    // The user is expected to implement a function
    // `cl_command_queue getCommandQueue()`
    cl_command_queue queue = getCommandQueue();

    cl_event waitEvent;

    cl_mem src = (cl_mem)pSource;
    cl_mem dst = pGLDestination->mId;

    glFinish();

    FORGE_OCL_CHECK(clEnqueueAcquireGLObjects(queue, 1, &dst, 0, NULL, &waitEvent),
                    "Failed in clEnqueueAcquireGLObjects");

    FORGE_OCL_CHECK(clWaitForEvents(1, &waitEvent),
                    "Failed in clWaitForEvents after clEnqueueAcquireGLObjects");

    FORGE_OCL_CHECK(clEnqueueCopyBuffer(queue, src, dst, 0, 0, pSize, 0, NULL, &waitEvent),
                    "Failed in clEnqueueCopyBuffer");

    FORGE_OCL_CHECK(clEnqueueReleaseGLObjects(queue, 1, &dst, 0, NULL, &waitEvent),
                    "Failed in clEnqueueReleaseGLObjects");

    FORGE_OCL_CHECK(clWaitForEvents(1, &waitEvent),
                    "Failed in clWaitForEvents after clEnqueueReleaseGLObjects");
}

#pragma GCC diagnostic pop

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif
