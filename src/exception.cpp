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
      mFuncName(pFuncName), mLineNumber(pLine), mErrCode(pErrCode)
{}

Error::Error(std::string pFuncName, const int pLine,
             std::string pMessage, ErrorCode pErrCode)
    : logic_error(pMessage),
      mFuncName(pFuncName), mLineNumber(pLine), mErrCode(pErrCode)
{}

const string& Error::functionName() const { return mFuncName; }

int Error::line() const { return mLineNumber; }

ErrorCode Error::err() const { return mErrCode; }

Error::~Error() throw() {}


TypeError::TypeError(const char * const pFuncName, const int pLine,
                     const int pIndex, const GLenum pType)
    : Error(pFuncName, pLine, "Invalid data type", FG_ERR_INVALID_TYPE),
      mArgIndex(pIndex),
      mErrTypeName(getName(pType)) /* TODO getName has to be defined */
{}

TypeError::TypeError(std::string pFuncName, const int pLine,
                     const int pIndex, const GLenum pType)
    : Error(pFuncName, pLine, "Invalid data type", FG_ERR_INVALID_TYPE),
      mArgIndex(pIndex),
      mErrTypeName(getName(pType)) /* TODO getName has to be defined */
{}

const string& TypeError::typeName() const { return mErrTypeName; }

int TypeError::argIndex() const { return mArgIndex; }

TypeError::~TypeError() throw() {}


ArgumentError::ArgumentError(const char * const pFuncName,
                             const int pLine,
                             const int pIndex,
                             const char * const pExpectString)
    : Error(pFuncName, pLine, "Invalid argument", FG_ERR_INVALID_ARG),
      mArgIndex(pIndex),
      mExpected(pExpectString)
{}

ArgumentError::ArgumentError(std::string pFuncName,
                             const int pLine,
                             const int pIndex,
                             std::string pExpectString)
    : Error(pFuncName, pLine, "Invalid argument", FG_ERR_INVALID_ARG),
      mArgIndex(pIndex),
      mExpected(pExpectString)
{}

const string& ArgumentError::expectedCondition() const { return mExpected; }

int ArgumentError::argIndex() const { return mArgIndex; }

ArgumentError::~ArgumentError() throw() {}


DimensionError::DimensionError(const char * const pFuncName,
                               const int pLine,
                               const int pIndex,
                               const char * const pExpectString)
    : Error(pFuncName, pLine, "Invalid dimension", FG_ERR_SIZE),
      mArgIndex(pIndex),
      mExpected(pExpectString)
{}

DimensionError::DimensionError(std::string pFuncName,
                               const int pLine,
                               const int pIndex,
                               std::string pExpectString)
    : Error(pFuncName, pLine, "Invalid dimension", FG_ERR_SIZE),
      mArgIndex(pIndex),
      mExpected(pExpectString)
{}

const string& DimensionError::expectedCondition() const { return mExpected; }

int DimensionError::argIndex() const { return mArgIndex; }

DimensionError::~DimensionError() throw() {}

}
