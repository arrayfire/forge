/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/util.hpp>
#include <cstdio>
#include <cstdlib>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <string>

#if defined(OS_WIN)
#include <Windows.h>
#endif

using std::string;

namespace forge {
namespace common {

float clampTo01(const float pValue)
{
    return (pValue < 0.0f ? 0.0f : (pValue>1.0f ? 1.0f : pValue));
}

#if defined(OS_WIN)
#include <windows.h>
#include <strsafe.h>
#include <vector>

void getFontFilePaths(std::vector<string>& pFiles,
                      const string& pDir, const string& pExt)
{
   WIN32_FIND_DATA ffd;
   LARGE_INTEGER filesize;
   TCHAR szDir[MAX_PATH];
   size_t length_of_arg;
   DWORD dwError=0;
   HANDLE hFind = INVALID_HANDLE_VALUE;

   // Check that the input path plus 3 is not longer than MAX_PATH.
   // Three characters are for the "\*" plus NULL appended below.
   StringCchLength(pDir.c_str(), MAX_PATH, &length_of_arg);

   if (length_of_arg > (MAX_PATH - 3)) {
        FG_ERROR("WIN API call: Directory path is too long", FG_ERR_FILE_NOT_FOUND);
   }

   //printf("\nTarget directory is %s\n\n", pDir.c_str());
   // Prepare string for use with FindFile functions.  First, copy the
   // string to a buffer, then append '\*' to the directory name.
   StringCchCopy(szDir, MAX_PATH, pDir.c_str());
   std::string wildcard = "\\*" + pExt;
   StringCchCat(szDir, MAX_PATH, wildcard.c_str());

   // Find the first file in the directory.
   hFind = FindFirstFile(szDir, &ffd);
   if (INVALID_HANDLE_VALUE == hFind) {
       FG_ERROR("WIN API call: file fetch in DIR failed", FG_ERR_FILE_NOT_FOUND);
   }

   // List all the files in the directory with some info about them.
   do {
      if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
         // It is a directory, skip the entry
         //_tprintf(TEXT("  %s   <DIR>\n"), ffd.cFileName);
      } else {
         filesize.LowPart = ffd.nFileSizeLow;
         filesize.HighPart = ffd.nFileSizeHigh;
         //_tprintf(TEXT("  %s   %ld bytes\n"), ffd.cFileName, filesize.QuadPart);
         pFiles.push_back(std::string(ffd.cFileName));
      }
   } while (FindNextFile(hFind, &ffd) != 0);

   dwError = GetLastError();
   if (dwError != ERROR_NO_MORE_FILES) {
        FG_ERROR("WIN API call: files fetch returned no files", FG_ERR_FILE_NOT_FOUND);
   }

   FindClose(hFind);
}
#endif

string clipPath(string path, string str)
{
    try {
        string::size_type pos = path.rfind(str);
        if(pos == string::npos) {
            return path;
        } else {
            return path.substr(pos);
        }
    } catch(...) {
        return path;
    }
}

string getEnvVar(const string &key)
{
#if defined(OS_WIN)
    DWORD bufSize = 32767; // limit according to GetEnvironment Variable documentation
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
    char * str = getenv(key.c_str());
    return str==NULL ? string("") : string(str);
#endif
}

string toString(const float pVal, const string pFormat)
{
    size_t len = std::to_string(pVal).length();
    std::unique_ptr<char> label(new char[len +1]);
    std::snprintf(label.get(), len, pFormat.c_str(), pVal);
    return std::string(label.get());
}

std::ostream& operator<<(std::ostream& pOut, const glm::mat4& pMat)
{
    const float* ptr = (const float*)glm::value_ptr(pMat);
    pOut<<"\n" << std::fixed;
    pOut<<ptr[0] << "\t" << ptr[1] << "\t" << ptr[2] << "\t" << ptr[3] << "\n";
    pOut<<ptr[4] << "\t" << ptr[5] << "\t" << ptr[6] << "\t" << ptr[7] << "\n";
    pOut<<ptr[8] << "\t" << ptr[9] << "\t" << ptr[10]<< "\t" << ptr[11]<< "\n";
    pOut<<ptr[12]<< "\t" << ptr[13]<< "\t" << ptr[14]<< "\t" << ptr[15]<< "\n";
    pOut<<"\n";
    return pOut;
}

glm::vec3 trackballPoint(const float pX, const float pY,
                         const float pWidth, const float pHeight)
{
    glm::vec3 P = glm::vec3(1.0*pX/pWidth*2 - 1.0, 1.0*pY/pHeight*2 - 1.0, 0);

    P.y = -P.y;
    float OP_squared = P.x * P.x + P.y * P.y;
    if (OP_squared <= 1*1) {
        P.z = sqrt(1*1 - OP_squared);
    } else {
        P.z = 0;
        P = glm::normalize(P);
    }
    return P;
}

}
}
