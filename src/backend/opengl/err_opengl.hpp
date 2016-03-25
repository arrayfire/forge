/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <fg/defines.h>

void glErrorCheck(const char *pMsg, const char* pFile, int pLine);
void glForceErrorCheck(const char *pMsg, const char* pFile, int pLine);

#define CheckGL(msg)      glErrorCheck     (msg, __FILE__, __LINE__)
#define ForceCheckGL(msg) glForceErrorCheck(msg, __FILE__, __LINE__)
