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
#include <iostream>
#include <stdexcept>

static const int MAX_ERR_STR_LEN = 1024;

namespace fg
{

class FGAPI Error : public std::logic_error
{
    char        mFuncName[MAX_ERR_STR_LEN];
    int         mLineNumber;
    ErrorCode   mErrCode;

    Error();

public:

    Error(const char * const pFuncName, int pLine, const char * const pMessage, ErrorCode pErrCode);

    const char* functionName() const;

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
    int   mArgIndex;
    char* mErrTypeName;

    TypeError();

public:

    TypeError(const char * const pFuncName,
              const int pLine,
              const int pIndex,
              const dtype pType);

    const char* typeName() const;

    int argIndex() const;

    ~TypeError() throw();
};

class FGAPI ArgumentError : public Error
{
    int   mArgIndex;
    char* mExpected;

    ArgumentError();

public:
    ArgumentError(const char * const pFuncName,
                  const int pLine,
                  const int pIndex,
                  const char * const pExpectString);

    const char* expectedCondition() const;

    int argIndex() const;

    ~ArgumentError() throw();
};

class FGAPI DimensionError : public Error
{
    int   mArgIndex;
    char* mExpected;

    DimensionError();

public:
    DimensionError(const char * const pFuncName,
                   const int pLine,
                   const int pIndex,
                   const char * const pExpectString);

    const char* expectedCondition() const;

    int argIndex() const;

    ~DimensionError() throw();
};

}
