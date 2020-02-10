Interop Helpers
===============

The user can include `fg/compute_copy.h` header to avoid writing necessary
compute backend's OpenGL interoperation code themselves.

Following defines should be declared prior to inclusion of this header
to use respective compute backend helpers.

- `USE_FORGE_CPU_COPY_HELPERS`
- `USE_FORGE_CUDA_COPY_HELPERS`
- `USE_FORGE_OPENCL_COPY_HELPERS`

.. doxygenstruct:: GfxHandle
    :project: Forge

.. doxygenenum:: BufferType
    :project: Forge

.. doxygentypedef:: ComputeResourceHandle
    :project: Forge

The following are the helper functions defined in this header. Dependending on which
`USE_FORGE_<compute>_COPY_HELPERS` macro is defined, the right set of helpers are chosen
at compile time.

.. code-block:: c

   void createGLBuffer(GfxHandle** pOut,
                       const unsigned pResourceId,
                       const BufferType pTarget);

   void releaseGLBuffer(GfxHandle* pHandle);

   void copyToGLBuffer(GfxHandle* pGLDestination,
                       ComputeResourceHandle  pSource,
                       const size_t pSize)

You can find the exact usage details of this header in the examples section.
