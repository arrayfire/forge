/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <fg/defines.h>

#include <stdexcept>
#include <string>
#include <cassert>
#include <vector>

void print_error(const std::string &msg);

fg_err processException();

#define CATCHALL                         \
    catch(...) {                         \
        return processException();       \
    }
