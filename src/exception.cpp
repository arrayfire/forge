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
#include <string.h>

using std::string;
using std::stringstream;
using std::cerr;

std::string getName(GLenum pType)
{
    // FIXME
    return std::string("test");
}

namespace fg
{

Error::Error(const char * const pFuncName, const int pLine,
             const char * const pMessage, ErrorCode pErrCode)
    : logic_error(pMessage),
      mFuncName(new char[strlen(pFuncName)+1]), mLineNumber(pLine), mErrCode(pErrCode)
{
    size_t len = strlen(pFuncName);
    mFuncName = new char[len + 1];
    strcpy_s(mFuncName, len, pFuncName);
    mFuncName[len] = '\0';
}

const char* const  Error::functionName() const { return mFuncName; }

int Error::line() const { return mLineNumber; }

ErrorCode Error::err() const { return mErrCode; }

Error::~Error() throw() {
    delete[] mFuncName;
}


TypeError::TypeError(const char * const pFuncName, const int pLine,
                     const int pIndex, const GLenum pType)
    : Error(pFuncName, pLine, "Invalid data type", FG_ERR_INVALID_TYPE), mArgIndex(pIndex)
{
    std::string str = getName(pType); /* TODO getName has to be defined */
    size_t len = str.length();
    mErrTypeName = new char[len + 1];
    strcpy_s(mErrTypeName, len, str.c_str());
    mErrTypeName[len] = '\0';
}

const char* const  TypeError::typeName() const { return mErrTypeName; }

int TypeError::argIndex() const { return mArgIndex; }

TypeError::~TypeError() throw() {
    delete[] mErrTypeName;
}


ArgumentError::ArgumentError(const char * const pFuncName,
                             const int pLine,
                             const int pIndex,
                             const char * const pExpectString)
    : Error(pFuncName, pLine, "Invalid argument", FG_ERR_INVALID_ARG), mArgIndex(pIndex)
{
    size_t len = strlen(pExpectString);
    mExpected = new char[len + 1];
    strcpy_s(mExpected, len, pExpectString);
    mExpected[len] = '\0';
}

const char* const ArgumentError::expectedCondition() const { return mExpected; }

int ArgumentError::argIndex() const { return mArgIndex; }

ArgumentError::~ArgumentError() throw() {
    delete[] mExpected;
}


DimensionError::DimensionError(const char * const pFuncName,
                               const int pLine,
                               const int pIndex,
                               const char * const pExpectString)
    : Error(pFuncName, pLine, "Invalid dimension", FG_ERR_SIZE), mArgIndex(pIndex)
{
    size_t len = strlen(pExpectString);
    mExpected = new char[len + 1];
    strcpy_s(mExpected, len,pExpectString);
    mExpected[len] = '\0';
}

const char* const DimensionError::expectedCondition() const { return mExpected; }

int DimensionError::argIndex() const { return mArgIndex; }

DimensionError::~DimensionError() throw() {
    delete[] mExpected;
}

}
