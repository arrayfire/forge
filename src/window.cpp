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

#include <fg/window.h>
#include <window.hpp>
#include <common.hpp>
#include <memory>

using namespace fg;

#define GLFW_THROW_ERROR(msg, err) \
    throw fg::Error("Window constructor", __LINE__, msg, err);

static GLEWContext* current = nullptr;

GLEWContext* glewGetContext()
{
    return current;
}

namespace internal
{

void MakeContextCurrent(const window_impl* pWindow)
{
    if (pWindow != NULL) {
        glfwMakeContextCurrent(pWindow->get());
        current = pWindow->glewContext();
    }
    CheckGL("End MakeContextCurrent");
}

//FIXME we have to take care of this error callback in
// multithreaded scenario
static void windowErrorCallback(int pError, const char* pDescription)
{
    fputs(pDescription, stderr);
}

window_impl::window_impl(int pWidth, int pHeight, const char* pTitle,
                        std::weak_ptr<window_impl> pWindow, const bool invisible)
    : mWidth(pWidth), mHeight(pHeight), mWindow(nullptr),
      mRows(0), mCols(0)
{
    glfwSetErrorCallback(windowErrorCallback);

    if (!glfwInit()) {
        std::cerr << "ERROR: GLFW wasn't able to initalize\n";
        GLFW_THROW_ERROR("glfw initilization failed", fg::FG_ERR_GL_ERROR)
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    /* we are currently avoiding compatibility profile to avoid
     * issues with Vertex Arrays Objects being not sharable among
     * contexts. Not being able to share VAOs resulted in some issues
     * for Forge renderable objects while being rendered
     * onto different windows(windows that share context) on the fly.
     * */
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    if (invisible)
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    else
        glfwWindowHint(GLFW_VISIBLE, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    // create glfw window
    // if pWindow is not null, then the window created in this
    // constructor call will share the context with pWindow
    GLFWwindow* temp = nullptr;
    if (auto observe = pWindow.lock()) {
        temp = glfwCreateWindow(pWidth, pHeight, pTitle, nullptr, observe->get());
    } else {
        temp = glfwCreateWindow(pWidth, pHeight, pTitle, nullptr, nullptr);
    }

    if (!temp) {
        std::cerr<<"Error: Could not Create GLFW Window!\n";
        GLFW_THROW_ERROR("glfw window creation failed", fg::FG_ERR_GL_ERROR)
    }
    mWindow = temp;

    // create glew context so that it will bind itself to windows
    if (auto observe = pWindow.lock()) {
        mGLEWContext = observe->glewContext();
    } else {
        mGLEWContext = new GLEWContext();
        if (mGLEWContext == NULL) {
            std::cerr<<"Error: Could not create GLEW Context!\n";
            glfwDestroyWindow(mWindow);
            GLFW_THROW_ERROR("GLEW context creation failed", fg::FG_ERR_GL_ERROR)
        }
    }

    // Set context (before glewInit())
    MakeContextCurrent(this);

    //GLEW Initialization - Must be done
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        char buffer[128];
        sprintf(buffer, "GLEW init failed: Error: %s\n", glewGetErrorString(err));
        glfwDestroyWindow(mWindow);
        GLFW_THROW_ERROR(buffer, fg::FG_ERR_GL_ERROR);
    }
    err = glGetError();
    if (err!=GL_NO_ERROR && err!=GL_INVALID_ENUM) {
        // ignore this error, as GLEW is known to
        // have this issue with 3.2+ core profiles
        // they are yet to fix this in GLEW
        ForceCheckGL("GLEW initilization failed");
        GLFW_THROW_ERROR("GLEW initilization failed", fg::FG_ERR_GL_ERROR)
    }

    glfwSetWindowUserPointer(mWindow, this);

    auto keyboardCallback = [](GLFWwindow* w, int a, int b, int c, int d)
    {
        static_cast<window_impl*>(glfwGetWindowUserPointer(w))->keyboardHandler(a, b, c, d);
    };

    glfwSetKeyCallback(mWindow, keyboardCallback);
#ifdef WINDOWS_OS
    mCxt = glfwGetWGLContext(mWindow);
    mDsp = GetDC(glfwGetWin32Window(mWindow));
#endif
#ifdef LINUX_OS
    mCxt = glfwGetGLXContext(mWindow);
    mDsp = glfwGetX11Display();
#endif
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
    CheckGL("End Window::Window");
}

window_impl::~window_impl()
{
    glfwDestroyWindow(mWindow);
}

void window_impl::setFont(const std::shared_ptr<font_impl>& pFont)
{
    mFont = pFont;
}

void window_impl::setTitle(const char* pTitle)
{
    CheckGL("Begin Window::setTitle");
    glfwSetWindowTitle(mWindow, pTitle);
    CheckGL("End Window::setTitle");
}

void window_impl::setPos(int pX, int pY)
{
    CheckGL("Begin Window::setPos");
    glfwSetWindowPos(mWindow, pX, pY);
    CheckGL("End Window::setPos");
}

void window_impl::setColorMap(fg::ColorMap cmap)
{
    switch(cmap) {
        case FG_DEFAULT:
            mColorMapUBO = mCMap->defaultMap();
            mUBOSize     = mCMap->defaultLen();
            break;
        case FG_SPECTRUM:
            mColorMapUBO = mCMap->spectrum();
            mUBOSize     = mCMap->spectrumLen();
            break;
        case FG_COLORS:
            mColorMapUBO = mCMap->colors();
            mUBOSize     = mCMap->colorsLen();
            break;
        case FG_REDMAP:
            mColorMapUBO = mCMap->red();
            mUBOSize     = mCMap->redLen();
            break;
        case FG_MOOD:
            mColorMapUBO = mCMap->mood();
            mUBOSize     = mCMap->moodLen();
            break;
        case FG_HEAT:
            mColorMapUBO = mCMap->heat();
            mUBOSize     = mCMap->heatLen();
            break;
        case FG_BLUEMAP:
            mColorMapUBO = mCMap->blue();
            mUBOSize     = mCMap->blueLen();
            break;
    }
}

void window_impl::keyboardHandler(int pKey, int scancode, int pAction, int pMods)
{
    if (pKey == GLFW_KEY_ESCAPE && pAction == GLFW_PRESS) {
        glfwSetWindowShouldClose(mWindow, GL_TRUE);
    }
}

ContextHandle window_impl::context() const
{
    return mCxt;
}

DisplayHandle window_impl::display() const
{
    return mDsp;
}

int window_impl::width() const
{
    return mWidth;
}

int window_impl::height() const
{
    return mHeight;
}

GLEWContext* window_impl::glewContext() const
{
    return mGLEWContext;
}

GLFWwindow* window_impl::get() const
{
    return mWindow;
}

const std::shared_ptr<colormap_impl>& window_impl::colorMapPtr() const
{
    return mCMap;
}

void window_impl::hide()
{
    glfwHideWindow(mWindow);
}

void window_impl::show()
{
    glfwShowWindow(mWindow);
}

bool window_impl::close()
{
    return glfwWindowShouldClose(mWindow) != 0;
}

void window_impl::draw(const std::shared_ptr<AbstractRenderable>& pRenderable)
{
    CheckGL("Begin drawImage");
    glfwMakeContextCurrent(mWindow);

    int wind_width, wind_height;
    glfwGetFramebufferSize(mWindow, &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);

    // clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);

    pRenderable->setColorMapUBOParams(mColorMapUBO, mUBOSize);
    pRenderable->render(0, 0, wind_width, wind_height);

    glfwSwapBuffers(mWindow);
    glfwPollEvents();
    CheckGL("End drawImage");
}

void window_impl::grid(int pRows, int pCols)
{
    mRows= pRows;
    mCols= pCols;

    int wind_width, wind_height;
    glfwGetFramebufferSize(mWindow, &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mCellWidth = wind_width / mCols;
    mCellHeight = wind_height / mRows;
}


void window_impl::draw(int pColId, int pRowId,
                        const std::shared_ptr<AbstractRenderable>& pRenderable,
                        const char* pTitle)
{
    CheckGL("Begin show(column, row)");
    glfwMakeContextCurrent(mWindow);

    float pos[2] = {0.0, 0.0};
    int c     = pColId;
    int r     = pRowId;
    int x_off = c * mCellWidth;
    int y_off = (mRows - 1 - r) * mCellHeight;

    /* following margins are tested out for various
     * aspect ratios and are working fine. DO NOT CHANGE.
     * */
    int top_margin = int(0.06f*mCellHeight);
    int bot_margin = int(0.02f*mCellHeight);
    int lef_margin = int(0.02f*mCellWidth);
    int rig_margin = int(0.02f*mCellWidth);
    // set viewport to render sub image
    glViewport(x_off + lef_margin, y_off + bot_margin, mCellWidth - 2 * rig_margin, mCellHeight - 2 * top_margin);
    glScissor(x_off + lef_margin, y_off + bot_margin, mCellWidth - 2 * rig_margin, mCellHeight - 2 * top_margin);
    glEnable(GL_SCISSOR_TEST);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);

    pRenderable->setColorMapUBOParams(mColorMapUBO, mUBOSize);
    pRenderable->render(x_off, y_off, mCellWidth, mCellHeight);

    glDisable(GL_SCISSOR_TEST);
    glViewport(x_off, y_off, mCellWidth, mCellHeight);

    if (pTitle!=NULL) {
        mFont->setOthro2D(mCellWidth, mCellHeight);
        pos[0] = mCellWidth / 3.0f;
        pos[1] = mCellHeight*0.92f;
        mFont->render(pos, RED, pTitle, 16);
    }

    CheckGL("End show(column, row)");
}

void window_impl::draw()
{
    CheckGL("Begin show");
    glfwSwapBuffers(mWindow);
    glfwPollEvents();
    CheckGL("End show");
}

}

namespace fg
{

Window::Window(int pWidth, int pHeight, const char* pTitle, const Window* pWindow, const bool invisible)
{
    if (pWindow == nullptr) {
        value = new internal::_Window(pWidth, pHeight, pTitle, nullptr, invisible);
    } else {
        value = new internal::_Window(pWidth, pHeight, pTitle, pWindow->get(), invisible);
    }
}

Window::~Window()
{
    delete value;
}

Window::Window(const Window& other)
{
    value = new internal::_Window(*other.get());
}

void Window::setFont(Font* pFont)
{
    value->setFont(pFont->get());
}

void Window::setTitle(const char* pTitle)
{
    value->setTitle(pTitle);
}

void Window::setPos(int pX, int pY)
{
    value->setPos(pX, pY);
}

void Window::setColorMap(ColorMap cmap)
{
    value->setColorMap(cmap);
}

ContextHandle Window::context() const
{
    return value->context();
}

DisplayHandle Window::display() const
{
    return value->display();
}

int Window::width() const
{
    return value->width();
}

int Window::height() const
{
    return value->height();
}

internal::_Window* Window::get() const
{
    return value;
}

void Window::hide()
{
    value->hide();
}

void Window::show()
{
    value->show();
}

bool Window::close()
{
    return value->close();
}

void Window::makeCurrent()
{
    value->makeCurrent();
}

void Window::draw(const Image& pImage)
{
    value->draw(pImage.get());
}

void Window::draw(const Plot& pPlot)
{
    value->draw(pPlot.get());
}

void Window::draw(const Histogram& pHist)
{
    value->draw(pHist.get());
}

void Window::grid(int pRows, int pCols)
{
    value->grid(pRows, pCols);
}

void Window::draw(int pColId, int pRowId, const Image& pImage, const char* pTitle)
{
    value->draw(pColId, pRowId, pImage.get(), pTitle);
}

void Window::draw(int pColId, int pRowId, const Plot& pPlot, const char* pTitle)
{
    value->draw(pColId, pRowId, pPlot.get(), pTitle);
}

void Window::draw(int pColId, int pRowId, const Histogram& pHist, const char* pTitle)
{
    value->draw(pColId, pRowId, pHist.get(), pTitle);
}

void Window::draw()
{
    value->draw();
}

}
