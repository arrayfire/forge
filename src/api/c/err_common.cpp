/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_common.hpp>
#include <type_util.hpp>
#include <string>
#include <iostream>
#include <sstream>

using std::string;
using std::stringstream;
using std::cerr;

AfgfxError::AfgfxError(const char * const funcName,
                 const int line,
                 const char * const message, afgfx_err err)
    : logic_error   (message),
      functionName  (funcName),
      lineNumber(line),
      error(err)
{}

AfgfxError::AfgfxError(string funcName,
                 const int line,
                 string message, afgfx_err err)
    : logic_error   (message),
      functionName  (funcName),
      lineNumber(line),
      error(err)
{}

const string&
AfgfxError::getFunctionName() const
{
    return functionName;
}

int
AfgfxError::getLine() const
{
    return lineNumber;
}

afgfx_err
AfgfxError::getError() const
{
    return error;
}

AfgfxError::~AfgfxError() throw() {}

TypeError::TypeError(const char * const  funcName,
                     const int line,
                     const int index, const GLenum type)
    : AfgfxError (funcName, line, "Invalid data type", AFGFX_ERR_INVALID_TYPE),
      argIndex(index),
      errTypeName(getName(type))
{}

const string& TypeError::getTypeName() const
{
    return errTypeName;
}

int TypeError::getArgIndex() const
{
    return argIndex;
}

ArgumentError::ArgumentError(const char * const  funcName,
                             const int line,
                             const int index,
                             const char * const  expectString)
    : AfgfxError(funcName, line, "Invalid argument", AFGFX_ERR_INVALID_ARG),
      argIndex(index),
      expected(expectString)
{

}

const string& ArgumentError::getExpectedCondition() const
{
    return expected;
}

int ArgumentError::getArgIndex() const
{
    return argIndex;
}


DimensionError::DimensionError(const char * const  funcName,
                             const int line,
                             const int index,
                             const char * const  expectString)
    : AfgfxError(funcName, line, "Invalid dimension", AFGFX_ERR_SIZE),
      argIndex(index),
      expected(expectString)
{

}

const string& DimensionError::getExpectedCondition() const
{
    return expected;
}

int DimensionError::getArgIndex() const
{
    return argIndex;
}


afgfx_err processException()
{
    stringstream    ss;
    afgfx_err          err= AFGFX_ERR_INTERNAL;

    try {
        throw;
    } catch (const DimensionError &ex) {

        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid dimension for argument " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        cerr << ss.str();
        err = AFGFX_ERR_SIZE;

    } catch (const ArgumentError &ex) {

        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid argument at index " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        cerr << ss.str();
        err = AFGFX_ERR_ARG;

    } catch (const TypeError &ex) {

        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n";

        cerr << ss.str();
        err = AFGFX_ERR_INVALID_TYPE;
    } catch (const AfgfxError &ex) {

        ss << "Internal error in " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << ex.what() << "\n";

        cerr << ss.str();
        err = ex.getError();
    } catch (...) {

        cerr << "Unknown error\n";
        err = AFGFX_ERR_UNKNOWN;
    }

    return err;
}

GLenum glErrorSkip(const char *msg, const char* file, int line)
{
#ifndef NDEBUG
    GLenum x = glGetError();
    if (x != GL_NO_ERROR) {
        printf("GL Error Skipped at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
    }
    return x;
#else
    return 0;
#endif
}

GLenum glErrorCheck(const char *msg, const char* file, int line)
{
// Skipped in release mode
#ifndef NDEBUG
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        printf("GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
        AFGFX_ERROR("Error in Graphics", AFGFX_ERR_GL_ERROR);
    }
    return x;
#else
    return 0;
#endif
}

GLenum glForceErrorCheck(const char *msg, const char* file, int line)
{
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        printf("GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
        AFGFX_ERROR("Error in Graphics", AFGFX_ERR_GL_ERROR);
    }
    return x;
}
