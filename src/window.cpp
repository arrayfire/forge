/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#include <common.hpp>
#include <fg/window.h>
#include <window.hpp>

#include <algorithm>
#include <memory>
#include <mutex>

using namespace fg;

static GLEWContext* current = nullptr;

GLEWContext* glewGetContext()
{
    return current;
}

/* following function is thread safe */
int getNextUniqueId()
{
    static int wndUnqIdTracker = 0;
    static std::mutex wndUnqIdMutex;

    std::lock_guard<std::mutex> lock(wndUnqIdMutex);
    return wndUnqIdTracker++;
}

namespace internal
{

void MakeContextCurrent(const window_impl* pWindow)
{
    CheckGL("Begin MakeContextCurrent");
    if (pWindow != NULL) {
        pWindow->get()->makeContextCurrent();
        current = pWindow->glewContext();
    }
    CheckGL("End MakeContextCurrent");
}

window_impl::window_impl(int pWidth, int pHeight, const char* pTitle,
                        std::weak_ptr<window_impl> pWindow, const bool invisible)
    : mID(getNextUniqueId())
{
    if (auto observe = pWindow.lock()) {
        mWindow = new wtk::Widget(pWidth, pHeight, pTitle, observe->get(), invisible);
    } else {
        /* when windows are not sharing any context, just create
         * a dummy wtk::Widget object and pass it on */
        mWindow = new wtk::Widget(pWidth, pHeight, pTitle, nullptr, invisible);
    }

    /* create glew context so that it will bind itself to windows */
    if (auto observe = pWindow.lock()) {
        mGLEWContext = observe->glewContext();
    } else {
        mGLEWContext = new GLEWContext();
        if (mGLEWContext == NULL) {
            std::cerr<<"Error: Could not create GLEW Context!\n";
            throw fg::Error("window_impl constructor", __LINE__,
                    "GLEW context creation failed", fg::FG_ERR_GL_ERROR);
        }
    }

    /* Set context (before glewInit()) */
    MakeContextCurrent(this);

    /* GLEW Initialization - Must be done */
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        char buffer[128];
        sprintf(buffer, "GLEW init failed: Error: %s\n", glewGetErrorString(err));
        throw fg::Error("window_impl constructor", __LINE__, buffer, fg::FG_ERR_GL_ERROR);
    }
    err = glGetError();
    if (err!=GL_NO_ERROR && err!=GL_INVALID_ENUM) {
        /* ignore this error, as GLEW is known to
         * have this issue with 3.2+ core profiles
         * they are yet to fix this in GLEW
         */
        ForceCheckGL("GLEW initilization failed");
        throw fg::Error("window_impl constructor", __LINE__,
                "GLEW initilization failed", fg::FG_ERR_GL_ERROR);
    }

    mCxt = mWindow->getGLContextHandle();
    mDsp = mWindow->getDisplayHandle();
    /* copy colormap shared pointer if
     * this window shares context with another window
     * */
    if (auto observe = pWindow.lock()) {
        mCMap = observe->colorMapPtr();
    } else {
        mCMap = std::make_shared<colormap_impl>();
    }

    /* set the colormap to default */
    mColorMapUBO = mCMap->defaultMap();
    mUBOSize = mCMap->defaultLen();
    glEnable(GL_MULTISAMPLE);

    std::vector<glm::mat4>& mats = mWindow->mMVPs;
    mats.resize(mWindow->mRows*mWindow->mCols);
    std::fill(mats.begin(), mats.end(), glm::mat4(1));

    CheckGL("End Window::Window");
}

window_impl::~window_impl()
{
    delete mWindow;
}

void window_impl::setFont(const std::shared_ptr<font_impl>& pFont)
{
    mFont = pFont;
}

void window_impl::setTitle(const char* pTitle)
{
    mWindow->setTitle(pTitle);
}

void window_impl::setPos(int pX, int pY)
{
    mWindow->setPos(pX, pY);
}

void window_impl::setSize(unsigned pW, unsigned pH)
{
    mWindow->setSize(pW, pH);
}

void window_impl::setColorMap(fg::ColorMap cmap)
{
    switch(cmap) {
        case FG_DEFAULT_MAP:
            mColorMapUBO = mCMap->defaultMap();
            mUBOSize     = mCMap->defaultLen();
            break;
        case FG_SPECTRUM_MAP:
            mColorMapUBO = mCMap->spectrum();
            mUBOSize     = mCMap->spectrumLen();
            break;
        case FG_COLORS_MAP:
            mColorMapUBO = mCMap->colors();
            mUBOSize     = mCMap->colorsLen();
            break;
        case FG_RED_MAP:
            mColorMapUBO = mCMap->red();
            mUBOSize     = mCMap->redLen();
            break;
        case FG_MOOD_MAP:
            mColorMapUBO = mCMap->mood();
            mUBOSize     = mCMap->moodLen();
            break;
        case FG_HEAT_MAP:
            mColorMapUBO = mCMap->heat();
            mUBOSize     = mCMap->heatLen();
            break;
        case FG_BLUE_MAP:
            mColorMapUBO = mCMap->blue();
            mUBOSize     = mCMap->blueLen();
            break;
    }
}

int window_impl::getID() const
{
    return mID;
}

long long  window_impl::context() const
{
    return mCxt;
}

long long  window_impl::display() const
{
    return mDsp;
}

int window_impl::width() const
{
    return mWindow->mWidth;
}

int window_impl::height() const
{
    return mWindow->mHeight;
}

GLEWContext* window_impl::glewContext() const
{
    return mGLEWContext;
}

const wtk::Widget* window_impl::get() const
{
    return mWindow;
}

const std::shared_ptr<colormap_impl>& window_impl::colorMapPtr() const
{
    return mCMap;
}

void window_impl::hide()
{
    mWindow->hide();
}

void window_impl::show()
{
    mWindow->show();
}

bool window_impl::close()
{
    return mWindow->close();
}

void window_impl::draw(const std::shared_ptr<AbstractRenderable>& pRenderable)
{
    CheckGL("Begin window_impl::draw");
    MakeContextCurrent(this);
    mWindow->resetCloseFlag();
    glViewport(0, 0, mWindow->mWidth, mWindow->mHeight);

    const glm::mat4& mvp = mWindow->mMVPs[0];
    // clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);

    // set colormap call is equivalent to noop for non-image renderables
    pRenderable->setColorMapUBOParams(mColorMapUBO, mUBOSize);
    pRenderable->render(mID, 0, 0, mWindow->mWidth, mWindow->mHeight, mvp);

    mWindow->swapBuffers();
    mWindow->pollEvents();
    CheckGL("End window_impl::draw");
}

void window_impl::grid(int pRows, int pCols)
{
    glViewport(0, 0, mWindow->mWidth, mWindow->mHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mWindow->mRows       = pRows;
    mWindow->mCols       = pCols;
    mWindow->mCellWidth  = mWindow->mWidth  / mWindow->mCols;
    mWindow->mCellHeight = mWindow->mHeight / mWindow->mRows;

    // resize mvp array for views to appropriate size
    std::vector<glm::mat4>& mats = mWindow->mMVPs;
    mats.resize(mWindow->mRows*mWindow->mCols);
    std::fill(mats.begin(), mats.end(), glm::mat4(1));
}

void window_impl::draw(int pColId, int pRowId,
                       const std::shared_ptr<AbstractRenderable>& pRenderable,
                       const char* pTitle)
{
    CheckGL("Begin draw(column, row)");
    MakeContextCurrent(this);
    mWindow->resetCloseFlag();

    float pos[2] = {0.0, 0.0};
    int c     = pColId;
    int r     = pRowId;
    int x_off = c * mWindow->mCellWidth;
    int y_off = (mWindow->mRows - 1 - r) * mWindow->mCellHeight;

    const glm::mat4& mvp = mWindow->mMVPs[r+c*mWindow->mRows];
    /* following margins are tested out for various
     * aspect ratios and are working fine. DO NOT CHANGE.
     * */
    int top_margin = int(0.06f*mWindow->mCellHeight);
    int bot_margin = int(0.02f*mWindow->mCellHeight);
    int lef_margin = int(0.02f*mWindow->mCellWidth);
    int rig_margin = int(0.02f*mWindow->mCellWidth);
    // set viewport to render sub image
    glViewport(x_off + lef_margin, y_off + bot_margin,
            mWindow->mCellWidth - 2 * rig_margin, mWindow->mCellHeight - 2 * top_margin);
    glScissor(x_off + lef_margin, y_off + bot_margin,
            mWindow->mCellWidth - 2 * rig_margin, mWindow->mCellHeight - 2 * top_margin);
    glEnable(GL_SCISSOR_TEST);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);

    // set colormap call is equivalent to noop for non-image renderables
    pRenderable->setColorMapUBOParams(mColorMapUBO, mUBOSize);
    pRenderable->render(mID, x_off, y_off, mWindow->mCellWidth, mWindow->mCellHeight, mvp);

    glDisable(GL_SCISSOR_TEST);
    glViewport(x_off, y_off, mWindow->mCellWidth, mWindow->mCellHeight);

    if (pTitle!=NULL) {
        mFont->setOthro2D(mWindow->mCellWidth, mWindow->mCellHeight);
        pos[0] = mWindow->mCellWidth / 3.0f;
        pos[1] = mWindow->mCellHeight*0.92f;
        mFont->render(mID, pos, RED, pTitle, 16);
    }

    CheckGL("End draw(column, row)");
}

void window_impl::swapBuffers()
{
    mWindow->swapBuffers();
    mWindow->pollEvents();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

}

namespace fg
{

Window::Window(int pWidth, int pHeight, const char* pTitle, const Window* pWindow, const bool invisible)
{
    if (pWindow == nullptr) {
        mValue = new internal::_Window(pWidth, pHeight, pTitle, nullptr, invisible);
    } else {
        mValue = new internal::_Window(pWidth, pHeight, pTitle, pWindow->get(), invisible);
    }
}

Window::~Window()
{
    delete mValue;
}

Window::Window(const Window& other)
{
    mValue = new internal::_Window(*other.get());
}

void Window::setFont(Font* pFont)
{
    mValue->setFont(pFont->get());
}

void Window::setTitle(const char* pTitle)
{
    mValue->setTitle(pTitle);
}

void Window::setPos(int pX, int pY)
{
    mValue->setPos(pX, pY);
}

void Window::setSize(unsigned pW, unsigned pH)
{
    mValue->setSize(pW, pH);
}

void Window::setColorMap(ColorMap cmap)
{
    mValue->setColorMap(cmap);
}

long long Window::context() const
{
    return mValue->context();
}

long long Window::display() const
{
    return mValue->display();
}

int Window::width() const
{
    return mValue->width();
}

int Window::height() const
{
    return mValue->height();
}

internal::_Window* Window::get() const
{
    return mValue;
}

void Window::hide()
{
    mValue->hide();
}

void Window::show()
{
    mValue->show();
}

bool Window::close()
{
    return mValue->close();
}

void Window::makeCurrent()
{
    mValue->makeCurrent();
}

void Window::draw(const Image& pImage, const bool pKeepAspectRatio)
{
    mValue->draw(pImage.get(), pKeepAspectRatio);
}

void Window::draw(const Chart& pChart)
{
    mValue->draw(pChart.get());
}

void Window::grid(int pRows, int pCols)
{
    mValue->grid(pRows, pCols);
}

void Window::draw(int pColId, int pRowId, const Image& pImage, const char* pTitle, const bool pKeepAspectRatio)
{
    mValue->draw(pColId, pRowId, pImage.get(), pTitle, pKeepAspectRatio);
}

void Window::draw(int pColId, int pRowId, const Chart& pChart, const char* pTitle)
{
    mValue->draw(pColId, pRowId, pChart.get(), pTitle);
}

void Window::swapBuffers()
{
    mValue->swapBuffers();
}

}
