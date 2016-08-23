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

#ifdef __cplusplus
#include <iostream>
#include <stdexcept>

namespace forge
{

class FGAPI Error : public std::exception
{
private:

    char        mMessage[1024];

    ErrorCode   mErrCode;

public:

    ErrorCode err() const { return mErrCode; }

    Error();

    Error(const char * const pMessage);

    Error(const char * const pFileName, int pLine, ErrorCode pErrCode);

    Error(const char * const pMessage, const char * const pFileName, int pLine, ErrorCode pErrCode);

    Error(const char * const pMessage, const char * const pFuncName,
          const char * const pFileName, int pLine, ErrorCode pErrCode);

    virtual ~Error() throw();

    virtual const char * what() const throw() { return mMessage; }

    friend inline std::ostream& operator<<(std::ostream &s, const Error &e)
    { return s << e.what(); }
};

} // namespace forge

#endif

#ifdef __cplusplus
extern "C" {
#endif

FGAPI void fg_get_last_error(char **msg, int *len);
FGAPI const char * fg_err_to_string(const fg_err err);

#ifdef __cplusplus
}
#endif
