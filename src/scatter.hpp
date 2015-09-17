/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <common.hpp>
#include <chart.hpp>
#include <plot.hpp>
#include <memory>
#include <map>

class plot_impl;

namespace internal
{

class scatter_impl : public plot_impl {
//   private:
//       /* plot points characteristics */
//       GLuint    mNumPoints;
//       GLenum    mDataType;
//       float     mLineColor[4];
//       fg::FGMarkerType mMarkerType;
//       /* OpenGL Objects */
//       GLuint    mMainVBO;
//       size_t    mMainVBOsize;
//       GLuint    mMarkerProgram;
//       /* shared variable index locations */
//       GLuint    mPointIndex;
//       GLuint    mMarkerTypeIndex;
//       GLuint    mSpriteTMatIndex;
//
//       std::map<int, GLuint> mVAOMap;
//
//       /* bind and unbind helper functions
//        * for rendering resources */
//       void bindResources(int pWindowId);
//       void unbindResources() const;
//       GLuint markerTypeIndex() const;
//       GLuint spriteMatIndex() const;
//
   public:
       scatter_impl(unsigned pNumPoints, fg::FGType pDataType, fg::FGMarkerType=fg::FG_NONE) : plot_impl(;
       ~scatter_impl();
//
//       void setColor(float r, float g, float b);
//       GLuint vbo() const;
//       size_t size() const;
//
//       void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);
};

}
