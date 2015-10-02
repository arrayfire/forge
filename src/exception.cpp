/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/exception.h>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string.h>

using std::string;
using std::stringstream;
using std::cerr;

std::string getName(fg::dtype pType)
{
    // FIXME
    return std::string("test");
}

void stringcopy(char* dest, const char* src, size_t len)
{
#ifdef OS_WIN
    strncpy_s(dest, MAX_ERR_STR_LEN, src, len);
#else
    strncpy(dest, src, len);
#endif
}

namespace fg
{

Error::Error(const char * const pFuncName, const int pLine,
             const char * const pMessage, ErrorCode pErrCode)
    : logic_error(pMessage),
      mLineNumber(pLine), mErrCode(pErrCode)
{
    size_t len = std::min(MAX_ERR_STR_LEN - 1, (int)strlen(pFuncName));
    stringcopy(mFuncName, pFuncName, len);
    mFuncName[len] = '\0';
}

const char* Error::functionName() const { return mFuncName; }

int Error::line() const { return mLineNumber; }

ErrorCode Error::err() const { return mErrCode; }

Error::~Error() throw() {}


TypeError::TypeError(const char * const pFuncName, const int pLine,
                     const int pIndex, const fg::dtype pType)
    : Error(pFuncName, pLine, "Invalid data type", FG_ERR_INVALID_TYPE), mArgIndex(pIndex)
{
    std::string str = getName(pType); /* TODO getName has to be defined */
    size_t len = std::min(MAX_ERR_STR_LEN - 1, (int)str.length());
    stringcopy(mErrTypeName, str.c_str(), len);
    mErrTypeName[len] = '\0';
}

const char* TypeError::typeName() const { return mErrTypeName; }

int TypeError::argIndex() const { return mArgIndex; }

TypeError::~TypeError() throw() {}


ArgumentError::ArgumentError(const char * const pFuncName,
                             const int pLine,
                             const int pIndex,
                             const char * const pExpectString)
    : Error(pFuncName, pLine, "Invalid argument", FG_ERR_INVALID_ARG), mArgIndex(pIndex)
{
    size_t len = std::min(MAX_ERR_STR_LEN - 1, (int)strlen(pExpectString));
    stringcopy(mExpected, pExpectString, len);
    mExpected[len] = '\0';
}

const char* ArgumentError::expectedCondition() const { return mExpected; }

int ArgumentError::argIndex() const { return mArgIndex; }

ArgumentError::~ArgumentError() throw() {}


DimensionError::DimensionError(const char * const pFuncName,
                               const int pLine,
                               const int pIndex,
                               const char * const pExpectString)
    : Error(pFuncName, pLine, "Invalid dimension", FG_ERR_SIZE), mArgIndex(pIndex)
{
    size_t len = std::min(MAX_ERR_STR_LEN - 1, (int)strlen(pExpectString));
    stringcopy(mExpected, pExpectString, len);
    mExpected[len] = '\0';
}

const char* DimensionError::expectedCondition() const { return mExpected; }

int DimensionError::argIndex() const { return mArgIndex; }

DimensionError::~DimensionError() throw() {}

}
