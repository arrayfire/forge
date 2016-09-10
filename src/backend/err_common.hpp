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
#include <defines.hpp>

#include <stdexcept>
#include <string>
#include <cassert>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Exception Classes
// Error, TypeError, ArgumentError, DimensionError
////////////////////////////////////////////////////////////////////////////////
class FgError : public std::logic_error
{
    std::string mFuncName;
    std::string mFileName;
    int mLineNumber;
    forge::ErrorCode mErrCode;
    FgError();

public:

    FgError(const char * const pFuncName,
            const char * const pFileName,
            const int pLineNumber,
            const char * const pMessage, forge::ErrorCode pErrCode);

    FgError(std::string pFuncName,
            std::string pFileName,
            const int pLineNumber,
            std::string pMessage, forge::ErrorCode pErrCode);

    const std::string&
    getFunctionName() const
    {
        return mFuncName;
    }

    const std::string&
    getFileName() const
    {
        return mFileName;
    }

    int getLineNumber() const
    {
        return mLineNumber;
    }

    forge::ErrorCode getError() const
    {
        return mErrCode;
    }

    virtual ~FgError() throw();
};

// TODO: Perhaps add a way to return supported types
class TypeError : public FgError
{
    int   mArgIndex;
    std::string mErrTypeName;

    TypeError();

public:

    TypeError(const char * const pFuncName,
              const char * const pFileName,
              const int pLine,
              const int pIndex,
              const forge::dtype pType);

    const std::string& getTypeName() const
    {
        return mErrTypeName;
    }

    int getArgIndex() const
    {
        return mArgIndex;
    }

    ~TypeError() throw() {}
};

class ArgumentError : public FgError
{
    int   mArgIndex;
    std::string mExpected;

    ArgumentError();

public:
    ArgumentError(const char * const pFuncName,
                  const char * const pFileName,
                  const int pLine,
                  const int pIndex,
                  const char * const pExpectString);

    const std::string& getExpectedCondition() const
    {
        return mExpected;
    }

    int getArgIndex() const
    {
        return mArgIndex;
    }

    ~ArgumentError() throw() {}
};

class DimensionError : public FgError
{
    int   mArgIndex;
    std::string mExpected;

    DimensionError();

public:
    DimensionError(const char * const pFuncName,
                   const char * const pFileName,
                   const int pLine,
                   const int pIndex,
                   const char * const pExpectString);

    const std::string& getExpectedCondition() const
    {
        return mExpected;
    }

    int getArgIndex() const
    {
        return mArgIndex;
    }

    ~DimensionError() throw() {}
};

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////
static const int MAX_ERR_SIZE = 1024;

std::string& getGlobalErrorString();

void print_error(const std::string &msg);

fg_err processException();

const char * getName(forge::dtype type);

////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////
#define DIM_ASSERT(INDEX, COND) do {                        \
        if((COND) == false) {                               \
            throw DimensionError(__PRETTY_FUNCTION__,       \
                                 __FG_FILENAME__, __LINE__, \
                                 INDEX, #COND);             \
        }                                                   \
    } while(0)

#define ARG_ASSERT(INDEX, COND) do {                        \
        if((COND) == false) {                               \
            throw ArgumentError(__PRETTY_FUNCTION__,        \
                                __FG_FILENAME__, __LINE__,  \
                                INDEX, #COND);              \
        }                                                   \
    } while(0)

#define TYPE_ERROR(INDEX, type) do {                        \
        throw TypeError(__PRETTY_FUNCTION__,                \
                        __FG_FILENAME__, __LINE__,          \
                        INDEX, type);                       \
    } while(0)                                              \


#define FG_ERROR(MSG, ERR_TYPE) do {                        \
        throw FgError(__PRETTY_FUNCTION__,                  \
                      __FG_FILENAME__, __LINE__,            \
                      MSG, ERR_TYPE);                       \
    } while(0)

#define FG_RETURN_ERROR(MSG, ERR_TYPE) do {                 \
        FgError err(__PRETTY_FUNCTION__,                    \
                    __FG_FILENAME__, __LINE__,              \
                    MSG, ERR_TYPE);                         \
        std::stringstream s;                                \
        s << "Error in " << err.getFunctionName() << "\n"   \
          << "In file " << err.getFileName()                \
          << ":" << err.getLine()  << "\n"                  \
          << err.what() << "\n";                            \
        print_error(s.str());                               \
        return ERR_TYPE;                                    \
    } while(0)

#define TYPE_ASSERT(COND) do {                              \
        if ((COND) == false) {                              \
            FG_ERROR("Type mismatch inputs",                \
                     FG_ERR_DIFF_TYPE);                     \
        }                                                   \
    } while(0)

#define FG_ASSERT(COND, MESSAGE)                            \
    assert(MESSAGE && COND)

#define CATCHALL                                            \
    catch(...) {                                            \
        return processException();                          \
    }

// Convert internal exception to external forge::Error
#define CATCH_INTERNAL_TO_EXTERNAL                          \
    catch(...) {                                            \
        fg_err __err = processException();                  \
        char *msg = NULL; fg_get_last_error(&msg, NULL);    \
        forge::Error ex(msg, __PRETTY_FUNCTION__,           \
                        __FG_FILENAME__, __LINE__, __err);  \
        delete [] msg;                                      \
        throw ex;                                           \
    }

#define FG_CHECK(fn) do {                                   \
        fg_err __err = fn;                                  \
        if (__err == FG_SUCCESS) break;                     \
        throw FgError(__PRETTY_FUNCTION__,                  \
                      __FG_FILENAME__, __LINE__,            \
                      "\n", __err);                         \
    } while(0)
