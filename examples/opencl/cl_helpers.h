/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <forge.h>
#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>

using namespace cl;


static const std::string NVIDIA_PLATFORM = "NVIDIA CUDA";
static const std::string AMD_PLATFORM = "AMD Accelerated Parallel Processing";
static const std::string INTEL_PLATFORM = "Intel(R) OpenCL";
static const std::string APPLE_PLATFORM = "Apple";

#if defined (OS_MAC)
static const std::string CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
static const std::string CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif


Platform getPlatform(std::string pName, cl_int &error)
{
    typedef std::vector<Platform>::iterator PlatformIter;

    Platform ret_val;
    error = 0;
    try {
        // Get available platforms
        std::vector<Platform> platforms;
        Platform::get(&platforms);
        int found = -1;
        for(PlatformIter it=platforms.begin(); it<platforms.end(); ++it) {
            std::string temp = it->getInfo<CL_PLATFORM_NAME>();
            if (temp==pName) {
                found = it - platforms.begin();
                std::cout<<"Found platform: "<<temp<<std::endl;
                break;
            }
        }
        if (found==-1) {
            // Going towards + numbers to avoid conflict with OpenCl error codes
            error = +1; // requested platform not found
        } else {
            ret_val = platforms[found];
        }
    } catch(Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
        error = err.err();
    }
    return ret_val;
}

Platform getPlatform()
{
    cl_int errCode;
    Platform plat = getPlatform(NVIDIA_PLATFORM, errCode);
    if (errCode != CL_SUCCESS) {
        Platform plat = getPlatform(AMD_PLATFORM, errCode);
        if (errCode != CL_SUCCESS) {
            Platform plat = getPlatform(INTEL_PLATFORM, errCode);
            if (errCode != CL_SUCCESS) {
                Platform plat = getPlatform(APPLE_PLATFORM, errCode);
                if (errCode != CL_SUCCESS) {
                    exit(255);
                } else {
                    return plat;
                }
            } else
                return plat;
        } else
            return plat;
    }
    return plat;
}

bool checkExtnAvailability(const Device &pDevice, std::string pName)
{
    bool ret_val = false;
    // find the extension required
    std::string exts = pDevice.getInfo<CL_DEVICE_EXTENSIONS>();
    std::stringstream ss(exts);
    std::string item;
    while (std::getline(ss,item,' ')) {
        if (item==pName) {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

