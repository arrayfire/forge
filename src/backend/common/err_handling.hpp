/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/defines.hpp>
#include <fg/exception.h>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace forge {
namespace common {
////////////////////////////////////////////////////////////////////////////////
// Exception Classes
// Error, TypeError, ArgumentError
////////////////////////////////////////////////////////////////////////////////
class FgError : public std::logic_error {
    std::string mFuncName;
    std::string mFileName;
    int mLineNumber;
    forge::ErrorCode mErrCode;
    FgError();

   public:
    FgError(const char* const pFuncName, const char* const pFileName,
            const int pLineNumber, const char* const pMessage,
            forge::ErrorCode pErrCode);

    FgError(std::string pFuncName, std::string pFileName, const int pLineNumber,
            std::string pMessage, forge::ErrorCode pErrCode);

    const std::string& getFunctionName() const { return mFuncName; }

    const std::string& getFileName() const { return mFileName; }

    int getLineNumber() const { return mLineNumber; }

    forge::ErrorCode getError() const { return mErrCode; }

    virtual ~FgError() throw();
};

// TODO: Perhaps add a way to return supported types
class TypeError : public FgError {
    int mArgIndex;
    std::string mErrTypeName;

    TypeError();

   public:
    TypeError(const char* const pFuncName, const char* const pFileName,
              const int pLine, const int pIndex, const forge::dtype pType);

    const std::string& getTypeName() const { return mErrTypeName; }

    int getArgIndex() const { return mArgIndex; }

    ~TypeError() throw() {}
};

class ArgumentError : public FgError {
    int mArgIndex;
    std::string mExpected;

    ArgumentError();

   public:
    ArgumentError(const char* const pFuncName, const char* const pFileName,
                  const int pLine, const int pIndex,
                  const char* const pExpectString);

    const std::string& getExpectedCondition() const { return mExpected; }

    int getArgIndex() const { return mArgIndex; }

    ~ArgumentError() throw() {}
};

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////
static const int MAX_ERR_SIZE = 1024;

std::string& getGlobalErrorString();

void print_error(const std::string& msg);

fg_err processException();

const char* getName(forge::dtype type);

////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////
#define ARG_ASSERT(INDEX, COND)                                                \
    do {                                                                       \
        if ((COND) == false) {                                                 \
            throw forge::common::ArgumentError(                                \
                __PRETTY_FUNCTION__, __FG_FILENAME__, __LINE__, INDEX, #COND); \
        }                                                                      \
    } while (0)

#define TYPE_ERROR(INDEX, type)                                              \
    do {                                                                     \
        throw forge::common::TypeError(__PRETTY_FUNCTION__, __FG_FILENAME__, \
                                       __LINE__, INDEX, type);               \
    } while (0)

#define FG_ERROR(MSG, ERR_TYPE)                                            \
    do {                                                                   \
        throw forge::common::FgError(__PRETTY_FUNCTION__, __FG_FILENAME__, \
                                     __LINE__, MSG, ERR_TYPE);             \
    } while (0)

#define TYPE_ASSERT(COND)                                       \
    do {                                                        \
        if ((COND) == false) {                                  \
            FG_ERROR("Type mismatch inputs", FG_ERR_DIFF_TYPE); \
        }                                                       \
    } while (0)

#define CATCHALL                                  \
    catch (...) {                                 \
        return forge::common::processException(); \
    }

}  // namespace common
}  // namespace forge
