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
#include <err_opengl.hpp>
#include <window_impl.hpp>

#include <algorithm>
#include <memory>
#include <mutex>

using namespace gl;
using namespace forge;

namespace forge
{

#ifdef USE_FREEIMAGE
#include <FreeImage.h>

class FI_Manager
{
    public:
    static bool initialized;
    FI_Manager()
    {
#ifdef FREEIMAGE_LIB
        FreeImage_Initialise();
#endif
        initialized = true;
    }

    ~FI_Manager()
    {
#ifdef FREEIMAGE_LIB
        FreeImage_DeInitialise();
#endif
    }
};

bool FI_Manager::initialized = false;

static void FI_Init()
{
    static FI_Manager manager = FI_Manager();
}

class FI_BitmapResource
{
public:
    explicit FI_BitmapResource(FIBITMAP * p) :
        pBitmap(p)
    {
    }

    ~FI_BitmapResource()
    {
        FreeImage_Unload(pBitmap);
    }
private:
    FIBITMAP * pBitmap;
};
#endif //USE_FREEIMAGE

/* following function is thread safe */
int getNextUniqueId()
{
    static int wndUnqIdTracker = 0;
    static std::mutex wndUnqIdMutex;

    std::lock_guard<std::mutex> lock(wndUnqIdMutex);
    return wndUnqIdTracker++;
}

static std::mutex initMutex;
static int initCallCount = -1;

void initWtkIfNotDone()
{
    std::lock_guard<std::mutex> lock(initMutex);

    initCallCount++;

    if (initCallCount==0)
        forge::wtk::initWindowToolkit();
}

void destroyWtkIfDone()
{
    std::lock_guard<std::mutex> lock(initMutex);

    initCallCount--;

    if (initCallCount==-1)
        forge::wtk::destroyWindowToolkit();
}

namespace opengl
{

void MakeContextCurrent(const window_impl* pWindow)
{
    if (pWindow != NULL) {
        pWindow->get()->makeContextCurrent();
        glbinding::Binding::useCurrentContext();
    }
}

window_impl::window_impl(int pWidth, int pHeight, const char* pTitle,
                        std::weak_ptr<window_impl> pWindow, const bool invisible)
    : mID(getNextUniqueId())
{
    initWtkIfNotDone();

    if (auto observe = pWindow.lock()) {
        mWindow = new wtk::Widget(pWidth, pHeight, pTitle, observe->get(), invisible);
    } else {
        /* when windows are not sharing any context, just create
         * a dummy wtk::Widget object and pass it on */
        mWindow = new wtk::Widget(pWidth, pHeight, pTitle, nullptr, invisible);
    }
    /* Set context (before glewInit()) */
    MakeContextCurrent(this);

    glbinding::Binding::useCurrentContext();
    glbinding::Binding::initialize(false); // lazy function pointer evaluation

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

    mWindow->resizePixelBuffers();

    /* set the colormap to default */
    mColorMapUBO = mCMap->cmapUniformBufferId(FG_COLOR_MAP_DEFAULT);
    mUBOSize = mCMap->cmapLength(FG_COLOR_MAP_DEFAULT);
    glEnable(GL_MULTISAMPLE);

    mWindow->resetViewMatrices();
    mWindow->resetOrientationMatrices();

    /* setup default window font */
    mFont = std::make_shared<font_impl>();
#if defined(OS_WIN)
    mFont->loadSystemFont("Calibri");
#else
    mFont->loadSystemFont("Vera");
#endif
    glEnable(GL_DEPTH_TEST);

    CheckGL("End Window::Window");
}

window_impl::~window_impl()
{
    delete mWindow;
    destroyWtkIfDone();
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

void window_impl::setColorMap(forge::ColorMap cmap)
{
    mColorMapUBO = mCMap->cmapUniformBufferId(cmap);

    mUBOSize = mCMap->cmapLength(cmap);
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

    const glm::mat4& viewMatrix = mWindow->getViewMatrix(std::make_tuple(1, 1, 0));
    const glm::mat4& orientMatrix = mWindow->getOrientationMatrix(std::make_tuple(1, 1, 0));

    // clear color and depth buffers
    glClearColor(WHITE[0], WHITE[1], WHITE[2], WHITE[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set colormap call is equivalent to noop for non-image renderables
    pRenderable->setColorMapUBOParams(mColorMapUBO, mUBOSize);
    pRenderable->render(mID, 0, 0, mWindow->mWidth, mWindow->mHeight,
                        viewMatrix, orientMatrix);

    mWindow->swapBuffers();
    mWindow->pollEvents();
    CheckGL("End window_impl::draw");
}

void window_impl::draw(const int pRows, const int pCols, const int pIndex,
                       const std::shared_ptr<AbstractRenderable>& pRenderable,
                       const char* pTitle)
{
    CheckGL("Begin draw(rows, columns, index)");
    MakeContextCurrent(this);
    mWindow->resetCloseFlag();

    const int cellWidth  = mWindow->mWidth/pCols;
    const int cellHeight = mWindow->mHeight/pRows;

    int c    = pIndex % pCols;
    int r    = pIndex / pCols;
    int xOff = c * cellWidth;
    int yOff = (pRows-1-r) * cellHeight;

    const glm::mat4& viewMatrix = mWindow->getViewMatrix(std::make_tuple(pRows, pCols, pIndex));
    const glm::mat4& orientMatrix = mWindow->getOrientationMatrix(std::make_tuple(pRows, pCols, pIndex));

    /* following margins are tested out for various
     * aspect ratios and are working fine. DO NOT CHANGE.
     * */
    int topCushionGap    = int(0.06f*cellHeight);
    int bottomCushionGap = int(0.02f*cellHeight);
    int leftCushionGap   = int(0.02f*cellWidth);
    int rightCushionGap  = int(0.02f*cellWidth);
    /* current view port */
    int x = xOff + leftCushionGap;
    int y = yOff + bottomCushionGap;
    int w = cellWidth  - leftCushionGap - rightCushionGap;
    int h = cellHeight - bottomCushionGap - topCushionGap;
    /* set viewport to render sub image */
    glViewport(x, y, w, h);
    glScissor(x, y, w, h);
    glEnable(GL_SCISSOR_TEST);

    /* set colormap call is equivalent to noop for non-image renderables */
    pRenderable->setColorMapUBOParams(mColorMapUBO, mUBOSize);
    pRenderable->render(mID, x, y, w, h, viewMatrix, orientMatrix);

    glDisable(GL_SCISSOR_TEST);
    glViewport(x, y, cellWidth, cellHeight);

    float pos[2] = {0.0, 0.0};
    if (pTitle!=NULL) {
        mFont->setOthro2D(cellWidth, cellHeight);
        pos[0] = cellWidth / 3.0f;
        pos[1] = cellHeight*0.94f;
        mFont->render(mID, pos, AF_BLUE, pTitle, 18);
    }

    CheckGL("End draw(rows, columns, index)");
}

void window_impl::swapBuffers()
{
    mWindow->swapBuffers();
    mWindow->pollEvents();
    // clear color and depth buffers
    glClearColor(WHITE[0], WHITE[1], WHITE[2], WHITE[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void window_impl::saveFrameBuffer(const char* pFullPath)
{
#ifdef USE_FREEIMAGE
    FI_Init();

    auto FIErrorHandler = [](FREE_IMAGE_FORMAT pOutputFIFormat, const char* pMessage) {
        printf("FreeImage Error Handler: %s\n", pMessage);
    };

    FreeImage_SetOutputMessage(FIErrorHandler);

    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(pFullPath);
    if (format == FIF_UNKNOWN) {
        format = FreeImage_GetFIFFromFilename(pFullPath);
    }
    if (format == FIF_UNKNOWN) {
        FG_ERROR("Freeimage: unrecognized image format", FG_ERR_FREEIMAGE_UNKNOWN_FORMAT);
    }

    if (!(format==FIF_BMP || format==FIF_PNG)) {
        FG_ERROR("Supports only bmp and png as of now", FG_ERR_FREEIMAGE_SAVE_FAILED);
    }

    uint w = mWindow->mWidth;
    uint h = mWindow->mHeight;
    uint c = 4;
    uint d = c * 8;

    FIBITMAP* bmp = FreeImage_Allocate(w, h, d);
    if (!bmp) {
        FG_ERROR("Freeimage: allocation failed", FG_ERR_FREEIMAGE_BAD_ALLOC);
    }

    FI_BitmapResource bmpUnloader(bmp);

    uint pitch = FreeImage_GetPitch(bmp);
    uchar* dst = FreeImage_GetBits(bmp);

    /* as glReadPixels was called using PBO earlier, hopefully
     * it was async call(which it should be unless vendor driver
     * is doing something fishy) and the transfer is over by now
     * */
    glBindBuffer(GL_PIXEL_PACK_BUFFER, mWindow->mFramePBO);

    uchar* src = (uchar*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

    if (src) {
        // copy data from mapped memory location
        uint w = mWindow->mWidth;
        uint h = mWindow->mHeight;
        uint i = 0;

        for (uint y = 0; y < h; ++y) {
            for (uint x = 0; x < w; ++x) {
                *(dst + x * c + FI_RGBA_RED  ) = (uchar) src[4*i+0]; // r
                *(dst + x * c + FI_RGBA_GREEN) = (uchar) src[4*i+1]; // g
                *(dst + x * c + FI_RGBA_BLUE ) = (uchar) src[4*i+2]; // b
                *(dst + x * c + FI_RGBA_ALPHA) = (uchar) src[4*i+3]; // a
                ++i;
            }
            dst += pitch;
        }

        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    int flags = 0;
    if (format == FIF_JPEG)
        flags = flags | JPEG_QUALITYSUPERB;

    if (!(FreeImage_Save(format,bmp, pFullPath, flags) == TRUE)) {
        FG_ERROR("FreeImage Save Failed", FG_ERR_FREEIMAGE_SAVE_FAILED);
    }
#else
    FG_ERROR("Freeimage is not configured to build", FG_ERR_NOT_CONFIGURED);
#endif //USE_FREEIMAGE
}

}
}
