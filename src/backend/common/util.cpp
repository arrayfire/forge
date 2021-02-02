/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_handling.hpp>
#include <common/util.hpp>
#include <fg/defines.h>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#if defined(OS_WIN)
#include <Windows.h>
#endif

using glm::vec2;
using glm::vec3;
using glm::vec4;
using std::make_pair;
using std::pair;
using std::string;

namespace forge {
namespace common {

float clampTo01(const float pValue) {
    return (pValue < 0.0f ? 0.0f : (pValue > 1.0f ? 1.0f : pValue));
}

#if defined(OS_WIN)
#include <strsafe.h>
#include <windows.h>
#include <vector>

void getFontFilePaths(std::vector<string>& pFiles, const string& pDir,
                      const string& pExt) {
    WIN32_FIND_DATA ffd;
    LARGE_INTEGER filesize;
    TCHAR szDir[MAX_PATH];
    size_t length_of_arg;
    DWORD dwError = 0;
    HANDLE hFind  = INVALID_HANDLE_VALUE;

    // Check that the input path plus 3 is not longer than MAX_PATH.
    // Three characters are for the "\*" plus NULL appended below.
    StringCchLength(pDir.c_str(), MAX_PATH, &length_of_arg);

    if (length_of_arg > (MAX_PATH - 3)) {
        FG_ERROR("WIN API call: Directory path is too long",
                 FG_ERR_FILE_NOT_FOUND);
    }

    // printf("\nTarget directory is %s\n\n", pDir.c_str());
    // Prepare string for use with FindFile functions.  First, copy the
    // string to a buffer, then append '\*' to the directory name.
    StringCchCopy(szDir, MAX_PATH, pDir.c_str());
    std::string wildcard = "\\*" + pExt;
    StringCchCat(szDir, MAX_PATH, wildcard.c_str());

    // Find the first file in the directory.
    hFind = FindFirstFile(szDir, &ffd);
    if (INVALID_HANDLE_VALUE == hFind) {
        FG_ERROR("WIN API call: file fetch in DIR failed",
                 FG_ERR_FILE_NOT_FOUND);
    }

    // List all the files in the directory with some info about them.
    do {
        if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            // It is a directory, skip the entry
            //_tprintf(TEXT("  %s   <DIR>\n"), ffd.cFileName);
        } else {
            filesize.LowPart  = ffd.nFileSizeLow;
            filesize.HighPart = ffd.nFileSizeHigh;
            //_tprintf(TEXT("  %s   %ld bytes\n"), ffd.cFileName,
            // filesize.QuadPart);
            pFiles.push_back(std::string(ffd.cFileName));
        }
    } while (FindNextFile(hFind, &ffd) != 0);

    dwError = GetLastError();
    if (dwError != ERROR_NO_MORE_FILES) {
        FG_ERROR("WIN API call: files fetch returned no files",
                 FG_ERR_FILE_NOT_FOUND);
    }

    FindClose(hFind);
}
#endif

string clipPath(string path, string str) {
    try {
        string::size_type pos = path.rfind(str);
        if (pos == string::npos) {
            return path;
        } else {
            return path.substr(pos);
        }
    } catch (...) { return path; }
}

string getEnvVar(const string& key) {
#if defined(OS_WIN)
    DWORD bufSize =
        32767;  // limit according to GetEnvironment Variable documentation
    string retVal;
    retVal.resize(bufSize);
    bufSize = GetEnvironmentVariable(key.c_str(), &retVal[0], bufSize);
    if (!bufSize) {
        return string("");
    } else {
        retVal.resize(bufSize);
        return retVal;
    }
#else
    char* str = getenv(key.c_str());
    return str == NULL ? string("") : string(str);
#endif
}

string toString(const float pVal, const string pFormat) {
    size_t len = std::to_string(pVal).length();
    std::unique_ptr<char> label(new char[len + 1]);
    std::snprintf(label.get(), len, pFormat.c_str(), pVal);
    return std::string(label.get());
}

std::ostream& operator<<(std::ostream& pOut, const glm::mat4& pMat) {
    const float* ptr = (const float*)glm::value_ptr(pMat);
    pOut << "\n" << std::fixed;
    pOut << ptr[0] << "\t" << ptr[1] << "\t" << ptr[2] << "\t" << ptr[3]
         << "\n";
    pOut << ptr[4] << "\t" << ptr[5] << "\t" << ptr[6] << "\t" << ptr[7]
         << "\n";
    pOut << ptr[8] << "\t" << ptr[9] << "\t" << ptr[10] << "\t" << ptr[11]
         << "\n";
    pOut << ptr[12] << "\t" << ptr[13] << "\t" << ptr[14] << "\t" << ptr[15]
         << "\n";
    pOut << "\n";
    return pOut;
}

pair<vec3, float> calcRotationFromArcBall(const vec2& lastPos,
                                          const vec2& currPos,
                                          const vec4& viewport) {
    auto project = [](const float pX, const float pY, const float pWidth,
                      const float pHeight) {
        glm::vec3 P     = glm::vec3((2.0f * pX) / pWidth - 1.0f,
                                (2.0f * pY) / pHeight - 1.0f, 0.0f);
        float xySqrdSum = P.x * P.x + P.y * P.y;
        float rSqrd     = (ARC_BALL_RADIUS * ARC_BALL_RADIUS);
        float rSqrdBy2  = rSqrd / 2.0f;
        // Project to Hyperbolic Sheet if Sum of X^2 and Y^2 is
        // greater than (RADIUS^2)/2 ; Otherwise to a sphere
        P.z = (xySqrdSum > rSqrdBy2 ? rSqrdBy2 / sqrt(xySqrdSum)
                                    : sqrt(rSqrd - xySqrdSum));
        return P;
    };
    auto ORG = vec2(viewport[0], viewport[1]);
    // Offset window position to viewport frame of reference
    auto p1  = lastPos - ORG;
    auto p2  = currPos - ORG;
    auto op1 = project(p1.x, p1.y, viewport[2], viewport[3]);
    auto op2 = project(p2.x, p2.y, viewport[2], viewport[3]);
    auto n1  = glm::normalize(op1);
    auto n2  = glm::normalize(op2);

    return make_pair(glm::cross(op2, op1), std::acos(glm::dot(n1, n2)));
}

}  // namespace common
}  // namespace forge
