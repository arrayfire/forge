/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <stdexcept>
#include <string>
#include <cassert>
#include <fw/defines.h>
#include <vector>

class FwError   : public std::logic_error
{
    std::string functionName;
    int lineNumber;
    fw_err error;
    FwError();

public:

    FwError(const char * const funcName,
            const int line,
            const char * const message, fw_err err);

    FwError(std::string funcName,
            const int line,
            std::string message, fw_err err);

    const std::string&
    getFunctionName() const;

    int getLine() const;

    fw_err getError() const;

    virtual ~FwError() throw();
};

// TODO: Perhaps add a way to return supported types
class TypeError : public FwError
{
    int argIndex;
    std::string errTypeName;
    TypeError();

public:

    TypeError(const char * const  funcName,
              const int line,
              const int index,
              const fw_dtype type);

    const std::string&
    getTypeName() const;

    int getArgIndex() const;

    ~TypeError() throw() {}
};

class ArgumentError : public FwError
{
    int argIndex;
    std::string expected;
    ArgumentError();

public:
    ArgumentError(const char * const funcName,
                   const int line,
                   const int index,
                   const char * const expectString);

    const std::string&
    getExpectedCondition() const;

    int getArgIndex() const;

    ~ArgumentError() throw(){}
};

class DimensionError : public FwError
{
    int argIndex;
    std::string expected;
    DimensionError();

public:
    DimensionError(const char * const funcName,
                   const int line,
                   const int index,
                   const char * const expectString);

    const std::string&
    getExpectedCondition() const;

    int getArgIndex() const;

    ~DimensionError() throw(){}
};

fw_err processException();

#define DIM_ASSERT(INDEX, COND) do {                    \
        if((COND) == false) {                           \
            throw DimensionError(__FILE__, __LINE__,    \
                                 INDEX, #COND);         \
        }                                               \
    } while(0)

#define ARG_ASSERT(INDEX, COND) do {                \
        if((COND) == false) {                       \
            throw ArgumentError(__FILE__, __LINE__, \
                                INDEX, #COND);      \
        }                                           \
    } while(0)

#define TYPE_ERROR(INDEX, type) do {            \
        throw TypeError(__FILE__, __LINE__,     \
                        INDEX, type);           \
    } while(0)                                  \


#define FW_ERROR(MSG, ERR_TYPE) do {            \
        throw FwError(__FILE__, __LINE__,       \
                      MSG, ERR_TYPE);           \
    } while(0)

#define FW_ASSERT(COND, MESSAGE)                \
    assert(MESSAGE && COND)

#define CATCHALL                                \
    catch(...) {                                \
        return processException();              \
    }

#define FW_CHECK(fn) do {                       \
        fw_err __err = fn;                      \
        if (__err == FW_SUCCESS) break;         \
        throw FwError(__FILE__, __LINE__,       \
                      "\n", __err);             \
    } while(0)

// GL Errors

GLenum glErrorSkip(const char *msg, const char* file, int line);
GLenum glErrorCheck(const char *msg, const char* file, int line);
GLenum glForceErrorCheck(const char *msg, const char* file, int line);

#define CheckGL(msg)      glErrorCheck     (msg, __FILE__, __LINE__)
#define ForceCheckGL(msg) glForceErrorCheck(msg, __FILE__, __LINE__)
#define CheckGLSkip(msg)  glErrorSkip      (msg, __FILE__, __LINE__)
