/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cstdint>

namespace forge {
namespace opengl {

class ShaderProgram {
   private:
    uint32_t mVertex;
    uint32_t mFragment;
    uint32_t mGeometry;
    uint32_t mProgram;

   public:
    ShaderProgram(const char* pVertShaderSrc, const char* pFragShaderSrc,
                  const char* pGeomShaderSrc = NULL);
    ~ShaderProgram();

    uint32_t getProgramId() const;
    uint32_t getUniformLocation(const char* pAttributeName);
    uint32_t getUniformBlockIndex(const char* pAttributeName);
    uint32_t getAttributeLocation(const char* pAttributeName);

    void bind();
    void unbind();
};

}  // namespace opengl
}  // namespace forge
