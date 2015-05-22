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

static std::shared_ptr<internal::window_impl> current;

struct NullDeleter { template<typename T> void operator()(T*) {} };

void MakeContextCurrent(internal::_Window* pWindow)
{
    if (pWindow != NULL)
    {
        glfwMakeContextCurrent((GLFWwindow*)pWindow->get());
        current = pWindow->wnd;
    }
    CheckGL("End MakeContextCurrent");
}

GLEWContext* glewGetContext()
{
    return current->mGLEWContext;
}

namespace internal
{

static void windowErrorCallback(int pError, const char* pDescription)
{
    fputs(pDescription, stderr);
}

#define GLFW_THROW_ERROR(msg, err) \
    throw fg::Error("Window constructor", __LINE__, msg, err);

_Window::_Window(int pWidth, int pHeight, const char* pTitle,
                const std::weak_ptr<_Window> pWindow, const bool invisible)
    : wnd(std::make_shared<window_impl>(pWidth, pHeight, pTitle))
{
    glfwSetErrorCallback(windowErrorCallback);

    if (!glfwInit()) {
        std::cerr << "ERROR: GLFW wasn't able to initalize\n";
        GLFW_THROW_ERROR("glfw initilization failed", fg::FG_ERR_GL_ERROR)
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (invisible)
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    else
        glfwWindowHint(GLFW_VISIBLE, GL_TRUE);
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
    wnd->mWindow = temp;

    // create glew context so that it will bind itself to windows
    if (auto observe = pWindow.lock()) {
        wnd->mGLEWContext = observe->glewContext();
    } else {
        wnd->mGLEWContext = new GLEWContext();
        if (wnd->mGLEWContext == NULL) {
            std::cerr<<"Error: Could not create GLEW Context!\n";
            glfwDestroyWindow(wnd->mWindow);
            GLFW_THROW_ERROR("GLEW context creation failed", fg::FG_ERR_GL_ERROR)
        }
    }

    // Set context (before glewInit())
    MakeContextCurrent(this);
    CheckGL("After MakeContextCurrent");

    //GLEW Initialization - Must be done
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        char buffer[128];
        sprintf(buffer, "GLEW init failed: Error: %s\n", glewGetErrorString(err));
        glfwDestroyWindow(wnd->mWindow);
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

    glfwSetWindowUserPointer(wnd->mWindow, this);

    auto keyboardCallback = [](GLFWwindow* w, int a, int b, int c, int d)
    {
        static_cast<_Window*>(glfwGetWindowUserPointer(w))->keyboardHandler(a, b, c, d);
    };

    glfwSetKeyCallback(wnd->mWindow, keyboardCallback);
#ifdef WINDOWS_OS
    wnd->mCxt = glfwGetWGLContext(wnd->mWindow);
    wnd->mDsp = GetDC(glfwGetWin32Window(wnd->mWindow));
#endif
#ifdef LINUX_OS
    wnd->mCxt = glfwGetGLXContext(wnd->mWindow);
    wnd->mDsp = glfwGetX11Display();
#endif

    CheckGL("End Window::Window");
}

void _Window::setTitle(const char* pTitle)
{
    CheckGL("Begin Window::setTitle");
    glfwSetWindowTitle(get(), pTitle);
    CheckGL("End Window::setTitle");
}

void _Window::setPos(int pX, int pY)
{
    CheckGL("Begin Window::setPos");
    glfwSetWindowPos(get(), pX, pY);
    CheckGL("End Window::setPos");
}

void _Window::keyboardHandler(int pKey, int scancode, int pAction, int pMods)
{
    if (pKey == GLFW_KEY_ESCAPE && pAction == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(get(), GL_TRUE);
    }
}

void _Window::draw(const fg::Image& pImage)
{
    CheckGL("Begin drawImage");
    MakeContextCurrent(this);

    int wind_width, wind_height;
    glfwGetWindowSize(get(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);

    // clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);

    pImage.render(0, 0, wind_width, wind_height);

    glfwSwapBuffers(get());
    glfwPollEvents();
    CheckGL("End drawImage");
}

void _Window::draw(const fg::Plot& pHandle)
{
    CheckGL("Begin draw(Plot)");
    MakeContextCurrent(this);

    int wind_width, wind_height;
    glfwGetWindowSize(get(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);

    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    pHandle.render(0, 0, wind_width, wind_height);

    glfwSwapBuffers(get());
    glfwPollEvents();
    CheckGL("End draw(Plot)");
}

void _Window::draw(const fg::Histogram& pHist)
{
    CheckGL("Begin draw(Histogram)");
    MakeContextCurrent(this);

    int wind_width, wind_height;
    glfwGetWindowSize(get(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    pHist.render(0, 0, wind_width, wind_height);

    glfwSwapBuffers(get());
    glfwPollEvents();
    CheckGL("End draw(Histogram)");
}

void _Window::grid(int pRows, int pCols)
{
    wnd->setGrid(pRows, pCols);

    int wind_width, wind_height;

    MakeContextCurrent(this);
    glfwGetWindowSize(get(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    wnd->setCellDims(wind_width / wnd->cols(), wind_height / wnd->rows());
}


template<typename T>
void _Window::draw(int pColId, int pRowId, const T& pRenderable, const char* pTitle)
{
    static const float RED[4] = {1.0, 0.0, 0.0, 1.0};
    float pos[2] = {0.0, 0.0};

    CheckGL("Begin show(column, row)");
    int c     = pColId;
    int r     = pRowId;
    int cellwidth = wnd->cellw();
    int cellheight = wnd->cellh();
    int x_off = c * cellwidth;
    int y_off = (wnd->rows() - 1 - r) * cellheight;

    /* following margins are tested out for various
     * aspect ratios and are working fine. DO NOT CHANGE.
     * */
    int top_margin = int(0.06f*cellheight);
    int bot_margin = int(0.02f*cellheight);
    int lef_margin = int(0.02f*cellwidth);
    int rig_margin = int(0.02f*cellwidth);
    // set viewport to render sub image
    glViewport(x_off + lef_margin, y_off + bot_margin, cellwidth - 2 * rig_margin, cellheight - 2 * top_margin);
    glScissor(x_off + lef_margin, y_off + bot_margin, cellwidth - 2 * rig_margin, cellheight - 2 * top_margin);
    glEnable(GL_SCISSOR_TEST);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);
    /* FIXME as of now, only fg::Image::render doesn't ask
     * for any input parameters */
    pRenderable.render(x_off, y_off, cellwidth, cellheight);
    glDisable(GL_SCISSOR_TEST);

    glViewport(x_off, y_off, cellwidth, cellheight);

	wnd->mFont->setOthro2D(cellwidth, cellheight);
    pos[0] = cellwidth / 3.0f;
    pos[1] = cellheight*0.92f;
    wnd->mFont->render(pos, RED, pTitle, 16);

    CheckGL("End show(column, row)");
}

#define INSTANTIATE_GRID_DRAW(T) \
    template void _Window::draw<T>(int pColId, int pRowId, const T& pRenderablePtr, const char* pTitle);

INSTANTIATE_GRID_DRAW(fg::Image);
INSTANTIATE_GRID_DRAW(fg::Histogram);
INSTANTIATE_GRID_DRAW(fg::Plot);

void _Window::draw()
{
    CheckGL("Begin show");
    glfwSwapBuffers(get());
    glfwPollEvents();
    CheckGL("End show");
}

}

namespace fg
{

Window::Window(int pWidth, int pHeight, const char* pTitle, const Window* pWindow, const bool invisible) {
    if (pWindow == nullptr) {
        std::shared_ptr<internal::_Window> other;
        value = new internal::_Window(pWidth, pHeight, pTitle, other, invisible);
    }
    else {
        std::shared_ptr<internal::_Window> other(pWindow->get(), NullDeleter());
        value = new internal::_Window(pWidth, pHeight, pTitle, other, invisible);
    }
}

Window::Window(const Window& other) {
}

void Window::setFont(Font* pFont) {
    value->setFont(pFont);
}

void Window::setTitle(const char* pTitle) {
    value->setTitle(pTitle);
}

void Window::setPos(int pX, int pY) {
    value->setPos(pX, pY);
}

ContextHandle Window::context() const {
    return value->context();
}

DisplayHandle Window::display() const {
    return value->display();
}

int Window::width() const {
    return value->width();
}

int Window::height() const {
    return value->height();
}

internal::_Window* Window::get() const {
    return value;
}

void Window::hide() {
    value->hide();
}

void Window::show() {
    value->show();
}

bool Window::close() {
    return value->close();
}

void Window::makeCurrent() {
    MakeContextCurrent(value);
}

void Window::draw(const Image& pImage) {
    value->draw(pImage);
}

void Window::draw(const Plot& pPlot) {
    value->draw(pPlot);
}
void Window::draw(const Histogram& pHist) {
    value->draw(pHist);
}

void Window::grid(int pRows, int pCols) {
    value->grid(pRows, pCols);
}

void Window::draw(int pColId, int pRowId, const Image& pImage, const char* pTitle) {
    value->draw(pColId, pRowId, pImage, pTitle);
}

void Window::draw(int pColId, int pRowId, const Plot& pPlot, const char* pTitle) {
    value->draw(pColId, pRowId, pPlot, pTitle);
}

void Window::draw(int pColId, int pRowId, const Histogram& pHist, const char* pTitle) {
    value->draw(pColId, pRowId, pHist, pTitle);
}

void Window::draw() {
    value->draw();
}

}
