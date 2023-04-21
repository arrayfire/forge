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

#include <common/err_handling.hpp>
#include <fg/update_buffer.h>
#include <gl_helpers.hpp>
#include <window_impl.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

using namespace forge;
using namespace forge::common;

using std::vector;

namespace forge {

#ifdef USE_FREEIMAGE
#include <FreeImage.h>

class FI_Manager {
   public:
    static bool initialized;
    FI_Manager() {
#ifdef FREEIMAGE_LIB
        FreeImage_Initialise();
#endif
        initialized = true;
    }

    ~FI_Manager() {
#ifdef FREEIMAGE_LIB
        FreeImage_DeInitialise();
#endif
    }
};

bool FI_Manager::initialized = false;

static void FI_Init() { static FI_Manager manager = FI_Manager(); }

class FI_BitmapResource {
   public:
    explicit FI_BitmapResource(FIBITMAP* p) : pBitmap(p) {}

    ~FI_BitmapResource() { FreeImage_Unload(pBitmap); }

   private:
    FIBITMAP* pBitmap;
};
#endif  // USE_FREEIMAGE

/* following function is thread safe */
int getNextUniqueId() {
    static int wndUnqIdTracker = 0;
    static std::mutex wndUnqIdMutex;

    std::lock_guard<std::mutex> lock(wndUnqIdMutex);
    return wndUnqIdTracker++;
}

static std::mutex initMutex;
static int initCallCount = -1;

void initWtkIfNotDone() {
    std::lock_guard<std::mutex> lock(initMutex);

    initCallCount++;

    if (initCallCount == 0) forge::wtk::initWindowToolkit();
}

void destroyWtkIfDone() {
    std::lock_guard<std::mutex> lock(initMutex);

    initCallCount--;

    if (initCallCount == -1) forge::wtk::destroyWindowToolkit();
}

namespace opengl {

void window_impl::prepArcBallObjects() {
    constexpr double angleStep =
        (2 * PI) / static_cast<double>(ARCBALL_CIRCLE_POINTS);

    vector<float> loop0(3 * (ARCBALL_CIRCLE_POINTS + 1)),
        loop1(3 * (ARCBALL_CIRCLE_POINTS + 1));
    int i = 0;
    while (i < ARCBALL_CIRCLE_POINTS) {
        loop0[3 * i + 0] = ARC_BALL_RADIUS * cos(i * angleStep);
        loop0[3 * i + 1] = ARC_BALL_RADIUS * sin(i * angleStep);
        loop0[3 * i + 2] = 0.0f;
        loop1[3 * i + 0] = 0.0f;
        loop1[3 * i + 1] = ARC_BALL_RADIUS * cos(i * angleStep);
        loop1[3 * i + 2] = ARC_BALL_RADIUS * sin(i * angleStep);
        i++;
    }
    // Since plot_impl's FG_PLOT_LINE is GL_LINE_STRIP
    // Add the first point again to make it a loop
    loop0[3 * i + 0] = ARC_BALL_RADIUS;
    loop0[3 * i + 1] = 0.0f;
    loop0[3 * i + 2] = 0.0f;
    loop1[3 * i + 0] = 0.0f;
    loop1[3 * i + 1] = ARC_BALL_RADIUS;
    loop1[3 * i + 2] = 0.0f;

    // Set Loop Colors
    mArcBallLoop0->setRanges(-1.0f, 1.0f, -1.0f, 1.0, -1.0f, 1.0f);
    mArcBallLoop1->setRanges(-1.0f, 1.0f, -1.0f, 1.0, -1.0f, 1.0f);
    mArcBallLoop0->setColor(AF_BLUE[0], AF_BLUE[1], AF_BLUE[2], AF_BLUE[3]);
    mArcBallLoop1->setColor(AF_BLUE[0], AF_BLUE[1], AF_BLUE[2], AF_BLUE[3]);

    // Update the respective plot_impl OpenGL buffers
    fg_update_vertex_buffer(mArcBallLoop0->vbo(), mArcBallLoop0->vboSize(),
                            loop0.data());
    fg_update_vertex_buffer(mArcBallLoop1->vbo(), mArcBallLoop1->vboSize(),
                            loop1.data());
}

window_impl::window_impl(int pWidth, int pHeight, const char* pTitle,
                         std::weak_ptr<window_impl> pWindow,
                         const bool invisible)
    : mID(getNextUniqueId()) {
    using widget_ptr = std::unique_ptr<wtk::Widget>;

    initWtkIfNotDone();

    if (auto observe = pWindow.lock()) {
        mWidget = widget_ptr(new wtk::Widget(pWidth, pHeight, pTitle,
                                             observe->get(), invisible));
    } else {
        /* when windows are not sharing any context, just create
         * a dummy wtk::Widget object and pass it on */
        mWidget = widget_ptr(
            new wtk::Widget(pWidth, pHeight, pTitle, widget_ptr(), invisible));
    }

    mWidget->makeContextCurrent();

    gladLoadGLLoader(mWidget->getProcAddr());

    mCxt = mWidget->getGLContextHandle();
    mDsp = mWidget->getDisplayHandle();
    /* copy colormap shared pointer if
     * this window shares context with another window
     * */
    if (auto observe = pWindow.lock()) {
        mCMap = observe->colorMapPtr();
    } else {
        mCMap = std::make_shared<colormap_impl>();
    }

    /* Create a plot for rendering Arc Ball orthogonal circles */
    mArcBallLoop0 = std::make_shared<plot_impl>(
        ARCBALL_CIRCLE_POINTS + 1, f32, FG_PLOT_LINE, FG_MARKER_NONE, 3, true);
    mArcBallLoop1 = std::make_shared<plot_impl>(
        ARCBALL_CIRCLE_POINTS + 1, f32, FG_PLOT_LINE, FG_MARKER_NONE, 3, true);
    prepArcBallObjects();

    /* set the colormap to default */
    mColorMapUBO = mCMap->cmapUniformBufferId(FG_COLOR_MAP_DEFAULT);
    mUBOSize     = mCMap->cmapLength(FG_COLOR_MAP_DEFAULT);
    glEnable(GL_MULTISAMPLE);

    mWidget->resetViewMatrices();
    mWidget->resetOrientationMatrices();

    /* setup default window font */
    mFont = std::make_shared<font_impl>();
#if defined(OS_WIN)
    mFont->loadSystemFont("Arial");
#else
    mFont->loadSystemFont("Vera");
#endif
    glEnable(GL_DEPTH_TEST);

    CheckGL("End Window::Window");
}

window_impl::~window_impl() {
    mCMap.reset();
    mFont.reset();
    mWidget.reset();
    destroyWtkIfDone();
}

void window_impl::makeContextCurrent() { mWidget->makeContextCurrent(); }

void window_impl::setFont(const std::shared_ptr<font_impl>& pFont) {
    mFont = pFont;
}

void window_impl::setTitle(const char* pTitle) { mWidget->setTitle(pTitle); }

void window_impl::setPos(int pX, int pY) { mWidget->setPos(pX, pY); }

void window_impl::setSize(unsigned pW, unsigned pH) {
    mWidget->setSize(pW, pH);
}

void window_impl::setColorMap(forge::ColorMap cmap) {
    mColorMapUBO = mCMap->cmapUniformBufferId(cmap);

    mUBOSize = mCMap->cmapLength(cmap);
}

int window_impl::getID() const { return mID; }

long long window_impl::context() const { return mCxt; }

long long window_impl::display() const { return mDsp; }

int window_impl::width() const { return mWidget->mWidth; }

int window_impl::height() const { return mWidget->mHeight; }

const std::unique_ptr<wtk::Widget>& window_impl::get() const { return mWidget; }

const std::shared_ptr<colormap_impl>& window_impl::colorMapPtr() const {
    return mCMap;
}

void window_impl::hide() { mWidget->hide(); }

void window_impl::show() { mWidget->show(); }

bool window_impl::close() { return mWidget->close(); }

void window_impl::draw(const std::shared_ptr<AbstractRenderable>& pRenderable) {
    CheckGL("Begin window_impl::draw");
    makeContextCurrent();
    mWidget->resetCloseFlag();
    glViewport(0, 0, mWidget->mWidth, mWidget->mHeight);

    const glm::mat4& viewMatrix =
        mWidget->getViewMatrix(std::make_tuple(1, 1, 0));
    const glm::mat4& orientMatrix =
        mWidget->getOrientationMatrix(std::make_tuple(1, 1, 0));

    // clear color and depth buffers
    glClearColor(WHITE[0], WHITE[1], WHITE[2], WHITE[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set colormap call is equivalent to noop for non-image renderables
    pRenderable->setColorMapUBOParams(mColorMapUBO, mUBOSize);
    pRenderable->render(mID, 0, 0, mWidget->mWidth, mWidget->mHeight,
                        viewMatrix, orientMatrix);
    // Render Arcball
    if (pRenderable->isRotatable() && mWidget->isBeingRotated()) {
        // TODO FIXME Figure out a better way to
        // render arc ball loops to include depth test for any
        // objects inside it
        glClear(GL_DEPTH_BUFFER_BIT);
        mArcBallLoop0->render(mID, 0, 0, mWidget->mWidth, mWidget->mHeight,
                              IDENTITY, orientMatrix);
        mArcBallLoop1->render(mID, 0, 0, mWidget->mWidth, mWidget->mHeight,
                              IDENTITY, orientMatrix);
    }
    mWidget->swapBuffers();
    mWidget->pollEvents();
    CheckGL("End window_impl::draw");
}

void window_impl::draw(const int pRows, const int pCols, const int pIndex,
                       const std::shared_ptr<AbstractRenderable>& pRenderable,
                       const char* pTitle) {
    CheckGL("Begin draw(rows, columns, index)");
    makeContextCurrent();
    mWidget->resetCloseFlag();

    const int cellWidth  = mWidget->mWidth / pCols;
    const int cellHeight = mWidget->mHeight / pRows;

    int c    = pIndex % pCols;
    int r    = pIndex / pCols;
    int xOff = c * cellWidth;
    int yOff = (pRows - 1 - r) * cellHeight;

    const glm::mat4& viewMatrix =
        mWidget->getViewMatrix(std::make_tuple(pRows, pCols, pIndex));
    const glm::mat4& orientMatrix =
        mWidget->getOrientationMatrix(std::make_tuple(pRows, pCols, pIndex));

    /* following margins are tested out for various
     * aspect ratios and are working fine. DO NOT CHANGE.
     * */
    int topCushionGap    = int(0.06f * cellHeight);
    int bottomCushionGap = int(0.02f * cellHeight);
    int leftCushionGap   = int(0.02f * cellWidth);
    int rightCushionGap  = int(0.02f * cellWidth);
    /* current view port */
    int x = xOff + leftCushionGap;
    int y = yOff + bottomCushionGap;
    int w = cellWidth - leftCushionGap - rightCushionGap;
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

    // Render Arcball if cursor position matches current cell
    glm::vec2 mpos = mWidget->getCursorPos();

    int wcx =
        static_cast<int>(std::floor(mpos[0] / static_cast<double>(cellWidth)));
    int wcy =
        static_cast<int>(std::floor(mpos[1] / static_cast<double>(cellHeight)));

    const bool isRotInCurrCell =
        (c == wcx && wcy == r && pRenderable->isRotatable());
    if (isRotInCurrCell && mWidget->isBeingRotated()) {
        // TODO FIXME Figure out a better way to
        // render arc ball loops to include depth test for any
        // objects inside it
        glClear(GL_DEPTH_BUFFER_BIT);
        mArcBallLoop0->render(mID, x, y, cellWidth, cellHeight, IDENTITY,
                              orientMatrix);
        mArcBallLoop1->render(mID, x, y, cellWidth, cellHeight, IDENTITY,
                              orientMatrix);
    }

    float pos[2] = {0.0, 0.0};
    if (pTitle != NULL) {
        mFont->setOthro2D(cellWidth, cellHeight);
        pos[0] = cellWidth / 3.0f;
        pos[1] = cellHeight * 0.94f;
        mFont->render(mID, pos, AF_BLUE, pTitle, 18);
    }

    CheckGL("End draw(rows, columns, index)");
}

void window_impl::swapBuffers() {
    mWidget->swapBuffers();
    mWidget->pollEvents();
    // clear color and depth buffers
    glClearColor(WHITE[0], WHITE[1], WHITE[2], WHITE[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void window_impl::saveFrameBuffer(const char* pFullPath) {
    this->makeContextCurrent();
#ifdef USE_FREEIMAGE
    FI_Init();

    auto FIErrorHandler = [](FREE_IMAGE_FORMAT pOutputFIFormat,
                             const char* pMessage) {
        printf("FreeImage Error Handler: %s\n", pMessage);
    };

    FreeImage_SetOutputMessage(FIErrorHandler);

    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(pFullPath);
    if (format == FIF_UNKNOWN) {
        format = FreeImage_GetFIFFromFilename(pFullPath);
    }
    if (format == FIF_UNKNOWN) {
        FG_ERROR("Freeimage: unrecognized image format",
                 FG_ERR_FREEIMAGE_UNKNOWN_FORMAT);
    }

    if (!(format == FIF_BMP || format == FIF_PNG)) {
        FG_ERROR("Supports only bmp and png as of now",
                 FG_ERR_FREEIMAGE_SAVE_FAILED);
    }

    uint32_t w = mWidget->mWidth;
    uint32_t h = mWidget->mHeight;
    uint32_t c = 4;
    uint32_t d = c * 8;

    FIBITMAP* bmp = FreeImage_Allocate(w, h, d);
    if (!bmp) {
        FG_ERROR("Freeimage: allocation failed", FG_ERR_FREEIMAGE_BAD_ALLOC);
    }

    FI_BitmapResource bmpUnloader(bmp);

    uint32_t pitch     = FreeImage_GetPitch(bmp);
    unsigned char* dst = FreeImage_GetBits(bmp);

    /* as glReadPixels was called using PBO earlier, hopefully
     * it was async call(which it should be unless vendor driver
     * is doing something fishy) and the transfer is over by now
     * */
    std::vector<GLubyte> pbuf(mWidget->mWidth * mWidget->mHeight * 4);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, mWidget->mWidth, mWidget->mHeight, GL_RGBA,
                 GL_UNSIGNED_BYTE, pbuf.data());
    uint32_t i = 0;

    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            *(dst + x * c + FI_RGBA_RED) = (unsigned char)pbuf[4 * i + 0];  // r
            *(dst + x * c + FI_RGBA_GREEN) =
                (unsigned char)pbuf[4 * i + 1];  // g
            *(dst + x * c + FI_RGBA_BLUE) =
                (unsigned char)pbuf[4 * i + 2];  // b
            *(dst + x * c + FI_RGBA_ALPHA) =
                (unsigned char)pbuf[4 * i + 3];  // a
            ++i;
        }
        dst += pitch;
    }
    int flags = 0;
    if (format == FIF_JPEG) flags = flags | JPEG_QUALITYSUPERB;

    if (!(FreeImage_Save(format, bmp, pFullPath, flags) == TRUE)) {
        FG_ERROR("FreeImage Save Failed", FG_ERR_FREEIMAGE_SAVE_FAILED);
    }
#else
    FG_ERROR("Freeimage is not configured to build", FG_ERR_NOT_CONFIGURED);
#endif  // USE_FREEIMAGE
}

}  // namespace opengl
}  // namespace forge
