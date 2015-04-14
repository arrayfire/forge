/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/defines.h>
#include <fg/exception.h>

#define FG_THROW(fn) do {                               \
        fg_err __err = fn;                                  \
        if (__err == FG_SUCCESS) break;                 \
        throw fg::exception(__FILE__, __LINE__, __err); \
    } while(0)

#define FG_THROW_MSG(__msg, __err) do {                         \
        if (__err == FG_SUCCESS) break;                         \
        throw fg::exception(__msg, __FILE__, __LINE__, __err);  \
    } while(0);
