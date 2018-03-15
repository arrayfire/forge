/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <string>

namespace forge
{
namespace common
{

inline std::string
clipPath(std::string path, std::string str)
{
    try {
        std::string::size_type pos = path.rfind(str);
        if(pos == std::string::npos) {
            return path;
        } else {
            return path.substr(pos);
        }
    } catch(...) {
        return path;
    }
}

#if defined(OS_WIN)
    #define __PRETTY_FUNCTION__ __FUNCSIG__
    #if _MSC_VER < 1900
        #define snprintf sprintf_s
    #endif
    #define STATIC_ static
    #define __FG_FILENAME__ (forge::common::clipPath(__FILE__, "src\\").c_str())
#else
    //#ifndef __PRETTY_FUNCTION__
    //    #define __PRETTY_FUNCTION__ __func__ // __PRETTY_FUNCTION__ Fallback
    //#endif
    #define STATIC_ inline
    #define __FG_FILENAME__ (forge::common::clipPath(__FILE__, "src/").c_str())
#endif

}
}
