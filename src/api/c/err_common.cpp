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

FgError::FgError(const char * const funcName,
                 const int line,
                 const char * const message, fg_err err)
    : logic_error   (message),
      functionName  (funcName),
      lineNumber(line),
      error(err)
{}

FgError::FgError(string funcName,
                 const int line,
                 string message, fg_err err)
    : logic_error   (message),
      functionName  (funcName),
      lineNumber(line),
      error(err)
{}

const string&
FgError::getFunctionName() const
{
    return functionName;
}

int
FgError::getLine() const
{
    return lineNumber;
}

fg_err
FgError::getError() const
{
    return error;
}

FgError::~FgError() throw() {}

TypeError::TypeError(const char * const  funcName,
                     const int line,
                     const int index, const GLenum type)
    : FgError (funcName, line, "Invalid data type", FG_ERR_INVALID_TYPE),
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
    : FgError(funcName, line, "Invalid argument", FG_ERR_INVALID_ARG),
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
    : FgError(funcName, line, "Invalid dimension", FG_ERR_SIZE),
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


fg_err processException()
{
    stringstream    ss;
    fg_err          err= FG_ERR_INTERNAL;

    try {
        throw;
    } catch (const DimensionError &ex) {

        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid dimension for argument " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        cerr << ss.str();
        err = FG_ERR_SIZE;

    } catch (const ArgumentError &ex) {

        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid argument at index " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        cerr << ss.str();
        err = FG_ERR_ARG;

    } catch (const TypeError &ex) {

        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n";

        cerr << ss.str();
        err = FG_ERR_INVALID_TYPE;
    } catch (const FgError &ex) {

        ss << "Internal error in " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << ex.what() << "\n";

        cerr << ss.str();
        err = ex.getError();
    } catch (...) {

        cerr << "Unknown error\n";
        err = FG_ERR_UNKNOWN;
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
        FG_ERROR("Error in Graphics", FG_ERR_GL_ERROR);
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
        FG_ERROR("Error in Graphics", FG_ERR_GL_ERROR);
    }
    return x;
}
