/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_handling.hpp>
#include <fg/exception.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

using std::cerr;
using std::string;
using std::stringstream;

namespace forge {

void stringcopy(char* dest, const char* src, size_t len) {
#if defined(OS_WIN)
    strncpy_s(dest, forge::common::MAX_ERR_SIZE, src, len);
#else
    std::strncpy(dest, src, len);
#endif
}

Error::Error() : mErrCode(FG_ERR_UNKNOWN) {
    stringcopy(mMessage, "Unknown Exception", sizeof(mMessage));
}

Error::Error(const char* const pMessage) : mErrCode(FG_ERR_UNKNOWN) {
    stringcopy(mMessage, pMessage, sizeof(mMessage));
    mMessage[sizeof(mMessage) - 1] = '\0';
}

Error::Error(const char* const pFileName, int pLine, ErrorCode pErrCode)
    : mErrCode(pErrCode) {
    std::snprintf(mMessage, sizeof(mMessage) - 1,
                  "Forge Exception (%s:%d):\nIn %s:%d",
                  fg_err_to_string(pErrCode), (int)pErrCode, pFileName, pLine);
    mMessage[sizeof(mMessage) - 1] = '\0';
}

Error::Error(const char* const pMessage, const char* const pFileName,
             const int pLine, ErrorCode pErrCode)
    : mErrCode(pErrCode) {
    std::snprintf(mMessage, sizeof(mMessage) - 1,
                  "Forge Exception (%s:%d):\n%s\nIn %s:%d",
                  fg_err_to_string(pErrCode), (int)pErrCode, pMessage,
                  pFileName, pLine);
    mMessage[sizeof(mMessage) - 1] = '\0';
}

Error::Error(const char* const pMessage, const char* const pFuncName,
             const char* const pFileName, const int pLine, ErrorCode pErrCode)
    : mErrCode(pErrCode) {
    std::snprintf(mMessage, sizeof(mMessage) - 1,
                  "Forge Exception (%s:%d):\n%sIn function %s\nIn file %s:%d",
                  fg_err_to_string(pErrCode), (int)pErrCode, pMessage,
                  pFuncName, pFileName, pLine);
    mMessage[sizeof(mMessage) - 1] = '\0';
}

Error::Error(const Error& error) {
    this->mErrCode = error.err();
    memcpy(this->mMessage, error.what(), 1024);
}

Error::~Error() throw() {}

}  // namespace forge
