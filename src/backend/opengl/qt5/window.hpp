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
#include <QWidget>
#include <glm/glm.hpp>

class QGLWidget;

namespace forge
{
namespace wtk
{

void initWindowToolkit();
void destroyWindowToolkit();

class EventFilter;

class Widget {
    private:
        QGLWidget *mWindow;
        std::unique_ptr<EventFilter> mEventFilter;
        bool          mClose;
        uint32_t      mWindowId;
        float         mLastXPos;
        float         mLastYPos;
        int           mButton;
//        SDL_Keycode   mMod;
        glm::vec3     mLastPos;
        bool mWindowCreated;


        MatrixHashMap mViewMatrices;
        MatrixHashMap mOrientMatrices;

        Widget();
        void initWindow(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow=nullptr);

        const glm::mat4 findTransform(const MatrixHashMap& pMap, const float pX, const float pY);

        const glm::mat4 getCellViewMatrix(const float pXPos, const float pYPos);

        const glm::mat4 getCellOrientationMatrix(const float pXPos, const float pYPos);

        void setTransform(MatrixHashMap& pMap, const float pX, const float pY, const glm::mat4 &pMat);

        void setCellViewMatrix(const float pXPos, const float pYPos, const glm::mat4& pMatrix);

        void setCellOrientationMatrix(const float pXPos, const float pYPos, const glm::mat4& pMatrix);

    public:
        /* public variables */
        int mWidth;     // Framebuffer width
        int mHeight;    // Framebuffer height
        bool mDockWidget;

        uint  mFramePBO;

        /* Constructors and methods */
        Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible);

        ~Widget();

        QWidget* getNativeHandle() const;

        void makeContextCurrent() const;

        long long getGLContextHandle();

        long long getDisplayHandle();

        bool getClose() const;

        void setTitle(const char* pTitle);

        void setPos(int pX, int pY);

        void setSize(unsigned pW, unsigned pH);

        void setClose(bool pClose);

        void swapBuffers();

        void hide();

        void show();

        bool close();

        void resetCloseFlag();

        void pollEvents();

        void resizePixelBuffers();

        const glm::mat4 getViewMatrix(const CellIndex& pIndex);

        const glm::mat4 getOrientationMatrix(const CellIndex& pIndex);

        void resetViewMatrices();

        void resetOrientationMatrices();

        bool eventFilter(QObject *object, QEvent *event);
};

}
}
