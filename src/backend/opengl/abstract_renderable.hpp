/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <string>

namespace forge {
namespace opengl {

/* Basic renderable class
 *
 * Any object that is renderable to a window should inherit from this
 * class.
 */
class AbstractRenderable {
   protected:
    /* OpenGL buffer objects */
    uint32_t mVBO;
    uint32_t mCBO;
    uint32_t mABO;
    size_t mVBOSize;
    size_t mCBOSize;
    size_t mABOSize;
    float mColor[4];
    float mRange[6];
    std::string mLegend;
    bool mIsPVCOn;
    bool mIsPVAOn;
    bool mIsInternalObject;

    AbstractRenderable()
        : mVBO(0)
        , mCBO(0)
        , mABO(0)
        , mVBOSize(0)
        , mCBOSize(0)
        , mABOSize(0)
        , mIsPVCOn(0)
        , mIsPVAOn(0)
        , mIsInternalObject(false) {
        mColor[0] = 0;
        mColor[1] = 0;
        mColor[2] = 0;
        mColor[3] = 0;

        mRange[0] = 0;
        mRange[1] = 0;
        mRange[2] = 0;
        mRange[3] = 0;
        mRange[4] = 0;
        mRange[5] = 0;
    }

   public:
    /* Getter functions for OpenGL buffer objects
     * identifiers and their size in bytes
     *
     *  vbo is for vertices
     *  cbo is for colors of those vertices
     *  abo is for alpha values for those vertices
     */
    uint32_t vbo() const { return mVBO; }
    uint32_t cbo() {
        mIsPVCOn = true;
        return mCBO;
    }
    uint32_t abo() {
        mIsPVAOn = true;
        return mABO;
    }
    size_t vboSize() const { return mVBOSize; }
    size_t cboSize() const { return mCBOSize; }
    size_t aboSize() const { return mABOSize; }

    /* Set color for rendering
     *
     * This method assumes, the color values are in
     * the range of [0, 1]
     */
    void setColor(const float pRed, const float pGreen, const float pBlue,
                  const float pAlpha) {
        mColor[0] = pRed;
        mColor[1] = pGreen;
        mColor[2] = pBlue;
        mColor[3] = pAlpha;
    }

    /* Get renderable solid color
     */
    void getColor(float& pRed, float& pGreen, float& pBlue, float& pAlpha) {
        pRed   = mColor[0];
        pGreen = mColor[1];
        pBlue  = mColor[2];
        pAlpha = mColor[3];
    }

    /* Set legend for rendering
     */
    void setLegend(const char* pLegend) { mLegend = std::string(pLegend); }

    /* Get legend string
     */
    const std::string& legend() const { return mLegend; }

    /* Set 3d world coordinate ranges
     *
     * This method is mostly used for charts and related renderables
     */
    void setRanges(const float pMinX, const float pMaxX, const float pMinY,
                   const float pMaxY, const float pMinZ, const float pMaxZ) {
        mRange[0] = pMinX;
        mRange[1] = pMaxX;
        mRange[2] = pMinY;
        mRange[3] = pMaxY;
        mRange[4] = pMinZ;
        mRange[5] = pMaxZ;
    }

    /* virtual function to set colormap, a derviced class might
     * use it or ignore it if it doesnt have a need for color maps.
     */
    virtual void setColorMapUBOParams(const uint32_t pUBO,
                                      const uint32_t pSize) {}

    /* render is a pure virtual function.
     *
     * @pWindowId is the window identifier
     * @pX is the X coordinate at which the currently bound viewport begins.
     * @pX is the Y coordinate at which the currently bound viewport begins.
     * @pViewPortWidth is the width of the currently bound viewport.
     * @pViewPortHeight is the height of the currently bound viewport.
     *
     * Any concrete class that inherits AbstractRenderable class needs to
     * implement this method to render their OpenGL objects to
     * the currently bound viewport of the Window bearing identifier pWindowId.
     *
     * @return nothing.
     */
    virtual void render(const int pWindowId, const int pX, const int pY,
                        const int pVPW, const int pVPH, const glm::mat4& pView,
                        const glm::mat4& pOrient) = 0;

    /*
     * Mark the renderable is for internal use for assistive help
     */
    inline void markAsInternalObject() { mIsInternalObject = true; }

    /*
     * Only rotatble renderables need to show 3d dimenionsional helper objects
     */
    virtual bool isRotatable() const = 0;
};

}  // namespace opengl
}  // namespace forge
