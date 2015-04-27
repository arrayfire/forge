/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <fg/defines.h>
#include <string>
#include <iostream>
#include <stdexcept>

namespace fg
{

class FGAPI Error : public std::logic_error
{
    std::string mFuncName;
    int         mLineNumber;
    ErrorCode   mErrCode;

    Error();

public:

    Error(const char * const pFuncName, int pLine, const char * const pMessage, ErrorCode pErrCode);

    Error(std::string pFuncName, int pLine, std::string pMessage, ErrorCode err);

    const std::string& functionName() const;

    int line() const;

    ErrorCode err() const;

    virtual ~Error() throw();

    friend inline std::ostream& operator<<(std::ostream &s, const Error &e) {
        return s << "@" << e.functionName() <<":"<< e.line()<<": "<<e.what()<<"("<<e.err()<<")"<<std::endl;
    }
};

// TODO: Perhaps add a way to return supported types
class FGAPI TypeError : public Error
{
    int         mArgIndex;
    std::string mErrTypeName;

    TypeError();

public:

    TypeError(const char * const pFuncName,
              const int pLine,
              const int pIndex,
              const GLenum pType);

    TypeError(std::string pFuncName,
              const int pLine,
              const int pIndex,
              const GLenum pType);

    const std::string& typeName() const;

    int argIndex() const;

    ~TypeError() throw();
};

class FGAPI ArgumentError : public Error
{
    int         mArgIndex;
    std::string mExpected;

    ArgumentError();

public:
    ArgumentError(const char * const pFuncName,
                  const int pLine,
                  const int pIndex,
                  const char * const pExpectString);

    ArgumentError(std::string pFuncName,
                  const int pLine,
                  const int pIndex,
                  std::string pExpectString);

    const std::string& expectedCondition() const;

    int argIndex() const;

    ~ArgumentError() throw();
};

class FGAPI DimensionError : public Error
{
    int         mArgIndex;
    std::string mExpected;

    DimensionError();

public:
    DimensionError(const char * const pFuncName,
                   const int pLine,
                   const int pIndex,
                   const char * const pExpectString);

    DimensionError(std::string pFuncName,
                   const int pLine,
                   const int pIndex,
                   std::string pExpectString);

    const std::string& expectedCondition() const;

    int argIndex() const;

    ~DimensionError() throw();
};

}
