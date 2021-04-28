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

#include <SDL2/SDL.h>
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
    SDL_Window* mWindow;
    SDL_GLContext mContext;
    SDL_Cursor* mDefaultCursor;
    SDL_Cursor* mRotationCursor;
    SDL_Cursor* mZoomCursor;
    SDL_Cursor* mMoveCursor;
    bool mClose;
    uint32_t mWindowId;
    glm::vec2 mLastPos;
    bool mRotationFlag;

    forge::common::MatrixHashMap mViewMatrices;
    forge::common::MatrixHashMap mOrientMatrices;

    Widget();

    const glm::vec4 getCellViewport(const glm::vec2& pos);
    const glm::mat4 findTransform(const forge::common::MatrixHashMap& pMap,
                                  const double pX, const double pY);
    const glm::mat4 getCellViewMatrix(const double pXPos, const double pYPos);
    const glm::mat4 getCellOrientationMatrix(const double pXPos,
                                             const double pYPos);
    void setTransform(forge::common::MatrixHashMap& pMap, const double pX,
                      const double pY, const glm::mat4& pMat);
    void setCellViewMatrix(const double pXPos, const double pYPos,
                           const glm::mat4& pMatrix);
    void setCellOrientationMatrix(const double pXPos, const double pYPos,
                                  const glm::mat4& pMatrix);

   public:
    /* public variables */
    int mWidth;   // Framebuffer width
    int mHeight;  // Framebuffer height

    /* Constructors and methods */
    Widget(int pWidth, int pHeight, const char* pTitle,
           const std::unique_ptr<Widget>& pWidget, const bool invisible);

    ~Widget();

    SDL_Window* getNativeHandle() const;

    void makeContextCurrent() const;

    long long getGLContextHandle();

    long long getDisplayHandle();

    GLADloadproc getProcAddr();

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

    const glm::mat4 getViewMatrix(const forge::common::CellIndex& pIndex);
    const glm::mat4 getOrientationMatrix(
        const forge::common::CellIndex& pIndex);
    void resetViewMatrices();
    void resetOrientationMatrices();

    inline bool isBeingRotated() const { return mRotationFlag; }

    glm::vec2 getCursorPos() const;
};

}  // namespace wtk
}  // namespace forge
