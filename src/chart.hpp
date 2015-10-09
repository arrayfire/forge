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
#include <vector>
#include <string>
#include <map>

#include <glm/glm.hpp>

namespace internal
{

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
        int       mTickCount;  /* should be an odd number always */
        int       mTickSize;
        int       mLeftMargin;
        int       mRightMargin;
        int       mTopMargin;
        int       mBottomMargin;
        /* chart axes ranges and titles */
        float    mXMax;
        float    mXMin;
        float    mYMax;
        float    mYMin;
        float    mZMax;
        float    mZMin;
        std::string mXTitle;
        std::string mYTitle;
        std::string mZTitle;
        /* OpenGL Objects */
        GLuint     mDecorVBO;
        GLuint     mBorderProgram;
        GLuint     mSpriteProgram;
        /* shader uniform variable locations */
        GLint     mBorderAttribPointIndex;
        GLint     mBorderUniformColorIndex;
        GLint     mBorderUniformMatIndex;
        GLint     mSpriteUniformMatIndex;
        GLint     mSpriteUniformTickcolorIndex;
        GLint     mSpriteUniformTickaxisIndex;
        /* VAO map to store a vertex array object
         * for each valid window context */
        std::map<int, GLuint> mVAOMap;

        /* rendering helper functions */
        void renderTickLabels(int pWindowId, unsigned w, unsigned h,
                std::vector<std::string> &texts,
                glm::mat4 &transformation, int coor_offset,
                bool useZoffset=true);

        /* virtual functions that has to be implemented by
         * dervied class: Chart2D, Chart3D */
        virtual void bindResources(int pWindowId) = 0;
        virtual void unbindResources() const = 0;
        virtual void pushTicktextCoords(float x, float y, float z=0.0) = 0;
        virtual void generateChartData() = 0;
        virtual void generateTickLabels() = 0;

    public:
        AbstractChart(int pLeftMargin, int pRightMargin, int pTopMargin, int pBottomMargin);
        virtual ~AbstractChart();

        void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin,
                           float pZmax=1, float pZmin=-1);
        void setAxesTitles(const char* pXTitle, const char* pYTitle, const char* pZTitle="Z-Axis");

        float xmax() const;
        float xmin() const;
        float ymax() const;
        float ymin() const;
        float zmax() const;
        float zmin() const;

        virtual GLuint vbo() const = 0;
        virtual size_t size() const = 0;
        virtual void renderChart(int pWindowId, int pX, int pY,
                                 int pViewPortWidth, int pViewPortHeight) = 0;
        /* Below is pure virtual function of AbstractRenderable */
        virtual void render(int pWindowId, int pX, int pY,
                            int pViewPortWidth, int pViewPortHeight) = 0;
};

class Chart2D : public AbstractChart {
    private:
        /* rendering helper functions that are derived
         * from AbstractRenderable base class
         * */
        void bindResources(int pWindowId);
        void unbindResources() const;
        void pushTicktextCoords(float x, float y, float z=0.0);
        void generateChartData();
        void generateTickLabels();

    public:
        Chart2D()
            :AbstractChart(68, 8, 8, 32) {
            generateChartData();
        }
        virtual ~Chart2D() {}

        void renderChart(int pWindowId, int pX, int pY,
                         int pViewPortWidth, int pViewPortHeight);

        /* Below pure virtual functions have to
         * be implemented by Concrete classes
         * which have Chart2D as base class
         * */
        virtual GLuint vbo() const = 0;
        virtual size_t size() const = 0;
        virtual void render(int pWindowId, int pX, int pY,
                            int pViewPortWidth, int pViewPortHeight) = 0;
};

class Chart3D : public AbstractChart {
    private:
        /* rendering helper functions that are derived
         * from AbstractRenderable base class
         * */
        void bindResources(int pWindowId);
        void unbindResources() const;
        void pushTicktextCoords(float x, float y, float z=0.0);
        void generateChartData();
        void generateTickLabels();

    public:
        Chart3D()
            :AbstractChart(32, 32, 32, 32) {
            generateChartData();
        }
        virtual ~Chart3D() {}

        void renderChart(int pWindowId, int pX, int pY,
                         int pViewPortWidth, int pViewPortHeight);

        virtual GLuint vbo() const = 0;
        virtual size_t size() const = 0;
        virtual void render(int pWindowId, int pX, int pY,
                            int pViewPortWidth, int pViewPortHeight) = 0;
};

}
