/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/defines.hpp>
#include <gl_helpers.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <memory>

/* the short form wtk stands for
 * Windowing Tool Kit */
namespace forge {
namespace wtk {

void initWindowToolkit();
void destroyWindowToolkit();

class Widget {
   private:
    GLFWwindow* mWindow;
    GLFWcursor* mRotationCursor;
    GLFWcursor* mZoomCursor;
    bool mClose;
    glm::vec2 mLastPos;
    int mButton;
    int mButtonAction;
    bool mRotationFlag;

    forge::common::MatrixHashMap mViewMatrices;
    forge::common::MatrixHashMap mOrientMatrices;

    Widget();

    const glm::mat4 findTransform(const forge::common::MatrixHashMap& pMap,
                                  const float pX, const float pY);
    const glm::mat4 getCellViewMatrix(const float pXPos, const float pYPos);
    const glm::mat4 getCellOrientationMatrix(const float pXPos,
                                             const float pYPos);
    void setTransform(forge::common::MatrixHashMap& pMap, const float pX,
                      const float pY, const glm::mat4& pMat);
    void setCellViewMatrix(const float pXPos, const float pYPos,
                           const glm::mat4& pMatrix);
    void setCellOrientationMatrix(const float pXPos, const float pYPos,
                                  const glm::mat4& pMatrix);

   public:
    /* public variables */
    int mWidth;   // Framebuffer width
    int mHeight;  // Framebuffer height

    /* Constructors and methods */
    Widget(int pWidth, int pHeight, const char* pTitle,
           const std::unique_ptr<Widget>& pWidget, const bool invisible);

    ~Widget();

    GLFWwindow* getNativeHandle() const;

    void makeContextCurrent() const;

    long long getGLContextHandle();

    long long getDisplayHandle();

    GLADloadproc getProcAddr();

    void setTitle(const char* pTitle);

    void setPos(int pX, int pY);

    void setSize(unsigned pW, unsigned pH);

    void swapBuffers();

    void hide();

    void show();

    bool close();

    void resetCloseFlag();

    void resizeHandler(int pWidth, int pHeight);

    void keyboardHandler(int pKey, int pScancode, int pAction, int pMods);

    void cursorHandler(float pXPos, float pYPos);

    void mouseButtonHandler(int pButton, int pAction, int pMods);

    void pollEvents();

    const glm::mat4 getViewMatrix(const forge::common::CellIndex& pIndex);
    const glm::mat4 getOrientationMatrix(
        const forge::common::CellIndex& pIndex);
    void resetViewMatrices();
    void resetOrientationMatrices();

    inline bool isBeingRotated() const { return mRotationFlag; }
};

}  // namespace wtk
}  // namespace forge
