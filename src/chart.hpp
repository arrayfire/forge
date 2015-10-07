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

class AbstractChart2D : public AbstractRenderable {
    private:
        /* internal class attributes for
        * drawing ticks on axes for plots*/
        std::vector<float> mTickTextX;
        std::vector<float> mTickTextY;
        std::vector<float> mTickTextZ;
        std::vector<std::string> mXText;
        std::vector<std::string> mYText;
        int       mTickCount;                       /* should be an odd number always */
        int       mTickSize;
        int       mLeftMargin;
        int       mRightMargin;
        int       mTopMargin;
        int       mBottomMargin;
        /* chart characteristics */
        float    mXMax;
        float    mXMin;
        float    mYMax;
        float    mYMin;
        std::string mXTitle;
        std::string mYTitle;
        /* OpenGL Objects */
        GLuint    mDecorVBO;
        GLuint    mBorderProgram;
        GLuint    mSpriteProgram;
        /* shader uniform variable locations */
        GLuint    mBorderPointIndex;
        GLuint    mBorderColorIndex;
        GLuint    mBorderMatIndex;
        GLuint    mSpriteMatIndex;
        GLuint    mSpriteTickcolorIndex;
        GLuint    mSpriteTickaxisIndex;
        /* VAO map to store a vertex array object
         * for each valid window context */
        std::map<int, GLuint> mVAOMap;

        void push_ticktext_points(float x, float y);
        void bindResources(int pWindowId);
        void unbindResources() const;

    protected:
        void bindBorderProgram() const;
        void unbindBorderProgram() const;
        GLuint rectangleVBO() const;
        GLuint borderProgramPointIndex() const;
        GLuint borderColorIndex() const;
        GLuint borderMatIndex() const;
        int tickSize() const;
        int leftMargin() const;
        int rightMargin() const;
        int bottomMargin() const;
        int topMargin() const;

        /* this function should be used after
        * setAxesLimits is called as this function uses
        * those values to generate the text markers that
        * are placed near the ticks on the axes */
        void setTickCount(int pTickCount);

    public:
        AbstractChart2D();
        virtual ~AbstractChart2D();

        void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin);
        void setXAxisTitle(const char* pTitle);
        void setYAxisTitle(const char* pTitle);

        float xmax() const;
        float xmin() const;
        float ymax() const;
        float ymin() const;
        void renderChart(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);

        virtual GLuint vbo() const = 0;
        virtual size_t size() const = 0;
        virtual void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight) = 0;
};

class AbstractChart3D : public AbstractRenderable {
    private:
        /* internal class attributes for
        * drawing ticks on axes for plots*/
        std::vector<float> mTickTextX;
        std::vector<float> mTickTextY;
        std::vector<float> mTickTextZ;
        std::vector<std::string> mXText;
        std::vector<std::string> mYText;
        std::vector<std::string> mZText;
        int       mTickCount;                       /* should be an odd number always */
        int       mTickSize;
        int       mLeftMargin;
        int       mRightMargin;
        int       mTopMargin;
        int       mBottomMargin;
        /* chart characteristics */
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
        GLuint    mDecorVBO;
        GLuint    mBorderProgram;
        GLuint    mSpriteProgram;
        /* shader uniform variable locations */
        GLuint    mBorderPointIndex;
        GLuint    mBorderColorIndex;
        GLuint    mBorderMatIndex;
        GLuint    mSpriteMatIndex;
        GLuint    mSpriteTickcolorIndex;
        GLuint    mSpriteTickaxisIndex;
        /* VAO map to store a vertex array object
         * for each valid window context */
        std::map<int, GLuint> mVAOMap;
        /* rendering helper functions */
        void push_ticktext_points(float x, float y, float z);
        void render_tickmarker_text(int pWindowId, unsigned w, unsigned h, std::vector<std::string> &texts, glm::mat4 &transformation, int coor_offset);

        void bindResources(int pWindowId);
        void unbindResources() const;

    protected:
        void bindBorderProgram() const;
        void unbindBorderProgram() const;
        GLuint rectangleVBO() const;
        GLuint borderProgramPointIndex() const;
        GLuint borderColorIndex() const;
        GLuint borderMatIndex() const;
        int tickSize() const;
        int leftMargin() const;
        int rightMargin() const;
        int bottomMargin() const;
        int topMargin() const;

        /* this function should be used after
        * setAxesLimits is called as this function uses
        * those values to generate the text markers that
        * are placed near the ticks on the axes */
        void setTickCount(int pTickCount);

    public:
        AbstractChart3D();
        virtual ~AbstractChart3D();

        void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin, float pZmax, float pZmin);
        void setXAxisTitle(const char* pTitle);
        void setYAxisTitle(const char* pTitle);
        void setZAxisTitle(const char* pTitle);

        float xmax() const;
        float xmin() const;
        float ymax() const;
        float ymin() const;
        float zmax() const;
        float zmin() const;
        void renderChart(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);

        virtual GLuint vbo() const = 0;
        virtual size_t size() const = 0;
        virtual void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight) = 0;
};
}
