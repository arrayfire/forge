/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/exception.h>

#include <common.hpp>
#include <err_opengl.hpp>

#include <sstream>
#include <iostream>

using namespace gl;

namespace forge
{
namespace opengl
{

void commonErrorCheck(const char *pMsg, const char* pFile, int pLine)
{
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        std::stringstream ss;
        ss << "GL Error at: "<< pFile << ":"<<pLine
           <<" Message: "<<pMsg<<" Error Code: "<< x << std::endl;
        FG_ERROR(ss.str().c_str(), FG_ERR_GL_ERROR);
    }
}

void glErrorCheck(const char *pMsg, const char* pFile, int pLine)
{
// Skipped in release mode
#ifndef NDEBUG
    commonErrorCheck(pMsg, pFile, pLine);
#endif
}

void glForceErrorCheck(const char *pMsg, const char* pFile, int pLine)
{
    commonErrorCheck(pMsg, pFile, pLine);
}

}
}
