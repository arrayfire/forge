/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/exception.h>

#include <err_common.hpp>
#include <util.hpp>

#include <string>
#include <sstream>

using namespace fg;

using std::string;
using std::stringstream;

std::string& getGlobalErrorString()
{
    static std::string global_error_string = std::string("");
    return global_error_string;
}

void print_error(const string &msg)
{
    std::string perr = getEnvVar("FG_PRINT_ERRORS");
    if(!perr.empty()) {
        if(perr != "0")
            fprintf(stderr, "%s\n", msg.c_str());
    }

    getGlobalErrorString() = msg;
}

fg_err processException()
{
    stringstream ss;
    fg_err err= FG_ERR_INTERNAL;

    try {
        throw;
    } catch (const TypeError &ex) {
        ss << ex << std::endl
           << "In function " << ex.functionName() << "\n"
           << "Invalid type for argument " << ex.argIndex() << "\n"
           << "Expects the type : "<< ex.typeName() << "\n";

        print_error(ss.str());
        err = FG_ERR_INVALID_TYPE;
    } catch (const ArgumentError &ex) {
        ss << ex << std::endl
           << "In function " << ex.functionName() << "\n"
           << "Invalid argument at index " << ex.argIndex() << "\n"
           << "Expected: " << ex.expectedCondition() << "\n";

        print_error(ss.str());
        err = FG_ERR_INVALID_ARG;
    } catch (const DimensionError &ex) {
        ss << ex << std::endl
           << "In function " << ex.functionName() << "\n"
           << "Invalid argument at index " << ex.argIndex() << "\n"
           << "Expected: " << ex.expectedCondition() << "\n";

        print_error(ss.str());
        err = FG_ERR_SIZE;
    } catch (...) {
        print_error(ss.str());
        err = FG_ERR_UNKNOWN;
    }

    return err;
}
