/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_common.hpp>
#include <fg/exception.h>

void commonErrorCheck(const char *pMsg, const char* pFile, int pLine)
{
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        char buffer[256];
        sprintf(buffer, "GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n",
                pFile, pLine, pMsg, x, gluErrorString(x));
        throw fg::Error(pFile, pLine, pMsg, fg::FG_ERR_GL_ERROR);
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
