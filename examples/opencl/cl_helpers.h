/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <forge.h>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include <CL/cl2.hpp>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <sstream>

using namespace cl;

#if defined(OS_MAC)
#include <OpenGL/OpenGL.h>
static const std::string CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
static const std::string CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif

bool checkGLInterop(const cl::Platform &plat, const cl::Device &pDevice,
                    const forge::Window &wnd) {
    bool ret_val = false;
    // find the extension required
    std::string exts = pDevice.getInfo<CL_DEVICE_EXTENSIONS>();
    std::stringstream ss(exts);
    std::string item;

    while (std::getline(ss, item, ' ')) {
        if (item == CL_GL_SHARING_EXT) {
            ret_val = true;
            break;
        }
    }

    if (!ret_val) return false;

#if !defined(OS_MAC)  // Check on Linux, Windows
        // Check if current OpenCL device is belongs to the OpenGL context

#if defined(OS_LNX)
    cl_context_properties cps[] = {CL_GL_CONTEXT_KHR,
                                   (cl_context_properties)wnd.context(),
                                   CL_GLX_DISPLAY_KHR,
                                   (cl_context_properties)wnd.display(),
                                   CL_CONTEXT_PLATFORM,
                                   (cl_context_properties)plat(),
                                   0};
#else /* OS_WIN */
    cl_context_properties cps[] = {CL_GL_CONTEXT_KHR,
                                   (cl_context_properties)wnd.context(),
                                   CL_WGL_HDC_KHR,
                                   (cl_context_properties)wnd.display(),
                                   CL_CONTEXT_PLATFORM,
                                   (cl_context_properties)plat(),
                                   0};
#endif

    // Load the extension
    // If cl_khr_gl_sharing is available, this function should be present
    // This has been checked earlier, it comes to this point only if it is found
    auto func =
        (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(
            plat(), "clGetGLContextInfoKHR");

    if (!func) return false;

    // Get all devices associated with opengl context
    std::vector<cl_device_id> devices(16);
    size_t ret = 0;
    cl_int err =
        func(cps, CL_DEVICES_FOR_GL_CONTEXT_KHR,
             devices.size() * sizeof(cl_device_id), devices.data(), &ret);

    if (err != CL_SUCCESS) return false;

    int num = (int)(ret / sizeof(cl_device_id));
    devices.resize(num);

    // Check if current device is present in the associated devices
    cl_device_id current_device = pDevice();
    auto res =
        std::find(std::begin(devices), std::end(devices), current_device);

    ret_val = res != std::end(devices);
#endif
    return ret_val;
}

cl::Context createCLGLContext(const forge::Window &wnd) {
    std::vector<cl::Platform> platforms;
    Platform::get(&platforms);

    for (auto platform : platforms) {
        std::vector<cl::Device> devices;

        try {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        } catch (const cl::Error &err) {
            if (err.err() != CL_DEVICE_NOT_FOUND) {
                throw;
            } else {
                continue;
            }
        }

        for (auto device : devices) {
            if (!checkGLInterop(platform, device, wnd)) continue;
#if defined(OS_MAC)
            CGLContextObj cgl_current_ctx = CGLGetCurrentContext();
            CGLShareGroupObj cgl_share_group =
                CGLGetShareGroup(cgl_current_ctx);

            cl_context_properties cps[] = {
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
                (cl_context_properties)cgl_share_group, 0};
#elif defined(OS_LNX)
            cl_context_properties cps[] = {CL_GL_CONTEXT_KHR,
                                           (cl_context_properties)wnd.context(),
                                           CL_GLX_DISPLAY_KHR,
                                           (cl_context_properties)wnd.display(),
                                           CL_CONTEXT_PLATFORM,
                                           (cl_context_properties)platform(),
                                           0};
#else /* OS_WIN */
            cl_context_properties cps[] = {CL_GL_CONTEXT_KHR,
                                           (cl_context_properties)wnd.context(),
                                           CL_WGL_HDC_KHR,
                                           (cl_context_properties)wnd.display(),
                                           CL_CONTEXT_PLATFORM,
                                           (cl_context_properties)platform(),
                                           0};
#endif
            std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>()
                      << std::endl;
            std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>()
                      << std::endl;
            return cl::Context(device, cps);
        }
    }

    throw std::runtime_error("No CL-GL sharing contexts found");
}

cl::CommandQueue queue;
cl::Context context;

cl_context getContext() { return context(); }

cl_command_queue getCommandQueue() { return queue(); }
