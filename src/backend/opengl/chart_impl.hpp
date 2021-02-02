/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <abstract_renderable.hpp>
#include <common/defines.hpp>
#include <shader_program.hpp>

#include <cstdint>
#include <map>

namespace forge {
namespace opengl {

class AbstractChart : public AbstractRenderable {
   protected:
    /* internal class attributes for
     * drawing ticks on axes for plots*/
    std::vector<float> mTickTextX;
    std::vector<float> mTickTextY;
    std::vector<float> mTickTextZ;
    std::vector<std::string> mXText;
    std::vector<std::string> mYText;
    std::vector<std::string> mZText;
    int mTickCount; /* should be an odd number always */
    float mTickSize;
    /* margin variables represent the % of current dimensions
     * and not the exact units of length */
    float mLeftMargin;
    float mRightMargin;
    float mTopMargin;
    float mBottomMargin;
    /* chart axes ranges and titles */
    bool mRenderAxes;
    std::string mXLabelFormat;
    float mXMax;
    float mXMin;
    std::string mYLabelFormat;
    float mYMax;
    float mYMin;
    std::string mZLabelFormat;
    float mZMax;
    float mZMin;
    std::string mXTitle;
    std::string mYTitle;
    std::string mZTitle;
    /* OpenGL Objects */
    uint32_t mDecorVBO;
    ShaderProgram mBorderProgram;
    ShaderProgram mSpriteProgram;
    /* shader uniform variable locations */
    uint32_t mBorderAttribPointIndex;
    uint32_t mBorderUniformColorIndex;
    uint32_t mBorderUniformMatIndex;
    uint32_t mSpriteUniformMatIndex;
    uint32_t mSpriteUniformTickcolorIndex;
    uint32_t mSpriteUniformTickaxisIndex;
    /* Chart legend position*/
    float mLegendX;
    float mLegendY;
    /* VAO map to store a vertex array object
     * for each valid window context */
    std::map<int, uint32_t> mVAOMap;
    /* list of renderables to be displayed on the chart*/
    std::vector<std::shared_ptr<AbstractRenderable>> mRenderables;

    /* rendering helper functions */
    inline float getTickStepSize(float minval, float maxval) const {
        return (maxval - minval) / (mTickCount - 1);
    }

    inline int getNumTicksC2E() const {
        /* Get # of ticks from center(0,0) to edge along axis
         * Excluding the center tick
         */
        return (mTickCount - 1) / 2;
    }

    inline float getLeftMargin(int pWidth) const {
        return pWidth * mLeftMargin;
    }

    inline float getRightMargin(int pWidth) const {
        return pWidth * mRightMargin;
    }

    inline float getBottomMargin(int pHeight) const {
        return pHeight * mBottomMargin;
    }

    inline float getTopMargin(int pHeight) const {
        return pHeight * mTopMargin;
    }

    inline float getTickSize() const { return mTickSize; }

    void renderTickLabels(const int pWindowId, const uint32_t pW,
                          const uint32_t pH,
                          const std::vector<std::string>& pTexts,
                          const int pFontSize, const glm::mat4& pTransformation,
                          const int pCoordsOffset,
                          const bool pUseZoffset = true) const;

    /* virtual functions that has to be implemented by
     * dervied class: chart2d_impl, chart3d_impl */
    virtual void bindResources(const int pWindowId)       = 0;
    virtual void unbindResources() const                  = 0;
    virtual void pushTicktextCoords(const float pX, const float pY,
                                    const float pZ = 0.0) = 0;
    virtual void generateChartData()                      = 0;
    virtual void generateTickLabels()                     = 0;

   public:
    AbstractChart(const float pLeftMargin, const float pRightMargin,
                  const float pTopMargin, const float pBottomMargin);
    virtual ~AbstractChart();

    void setAxesVisibility(const bool isVisible = true);

    void setAxesTitles(const char* pXTitle, const char* pYTitle,
                       const char* pZTitle);

    void setAxesLimits(const float pXmin, const float pXmax, const float pYmin,
                       const float pYmax, const float pZmin, const float pZmax);

    void setAxesLabelFormat(const std::string& pXFormat,
                            const std::string& pYFormat,
                            const std::string& pZFormat);

    void getAxesLimits(float* pXmin, float* pXmax, float* pYmin, float* pYmax,
                       float* pZmin, float* pZmax);

    void setLegendPosition(const float pX, const float pY);

    float xmax() const;
    float xmin() const;
    float ymax() const;
    float ymin() const;
    float zmax() const;
    float zmin() const;

    void addRenderable(const std::shared_ptr<AbstractRenderable> pRenderable);
};

class chart2d_impl : public AbstractChart {
   private:
    /* rendering helper functions that are derived
     * from AbstractRenderable base class
     * */
    void bindResources(const int pWindowId);
    void unbindResources() const;
    void pushTicktextCoords(const float x, const float y, const float z = 0.0);
    void generateChartData();
    void generateTickLabels();

   public:
    chart2d_impl();

    virtual ~chart2d_impl() {}

    void render(const int pWindowId, const int pX, const int pY, const int pVPW,
                const int pVPH, const glm::mat4& pView,
                const glm::mat4& pOrient);

    bool isRotatable() const { return false; }
};

class chart3d_impl : public AbstractChart {
   private:
    /* rendering helper functions that are derived
     * from AbstractRenderable base class
     * */
    void bindResources(const int pWindowId);
    void unbindResources() const;
    void pushTicktextCoords(const float x, const float y, const float z = 0.0);
    void generateChartData();
    void generateTickLabels();

   public:
    chart3d_impl();

    virtual ~chart3d_impl() {}

    void render(const int pWindowId, const int pX, const int pY, const int pVPW,
                const int pVPH, const glm::mat4& pView,
                const glm::mat4& pOrient);

    bool isRotatable() const { return true; }
};

}  // namespace opengl
}  // namespace forge
