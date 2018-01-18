/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <common.hpp>
#include <qt5/window.hpp>
#include <gl_native_handles.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtGui/QResizeEvent>

#include <QtOpenGL/QGLWidget>

#include "dockWrapper.hpp"

using namespace gl;

using glm::rotate;
using glm::translate;
using glm::scale;

#define SDL_THROW_ERROR(msg, err) \
    FG_ERROR("Window constructor "#msg,err)

namespace forge
{
namespace wtk
{

QApplication *qtApplication=nullptr;
QMainWindow *qtMainWindow=nullptr;

class EventFilter:public QObject
{
public:
    EventFilter(Widget *parent):parent(parent){}

    bool eventFilter(QObject *object, QEvent *event)
    {
        return parent->eventFilter(object, event);
    }

private:
    Widget *parent;

};

void initWindowToolkit()
{
}

void destroyWindowToolkit()
{
    if(qtApplication!=nullptr)
    {
        qtApplication->closeAllWindows();
        delete qtApplication;
    }
}

void setupApplication()
{
    int argc=1;
    char **argv=nullptr;

    qtApplication=new QApplication(argc, argv);
    qtMainWindow=new QMainWindow();
}

const glm::mat4 Widget::findTransform(const MatrixHashMap& pMap, const float pX, const float pY)
{
    for (auto it: pMap) {
        const CellIndex& idx = it.first;
        const glm::mat4& mat  = it.second;

        const int rows = std::get<0>(idx);
        const int cols = std::get<1>(idx);

        const int cellWidth  = mWidth/cols;
        const int cellHeight = mHeight/rows;

        const int x = int(pX) / cellWidth;
        const int y = int(pY) / cellHeight;
        const int i = x + y * cols;
        if (i==std::get<2>(idx)) {
            return mat;
        }
    }

    return IDENTITY;
}

const glm::mat4 Widget::getCellViewMatrix(const float pXPos, const float pYPos)
{
    return findTransform(mViewMatrices, pXPos, pYPos);
}

const glm::mat4 Widget::getCellOrientationMatrix(const float pXPos, const float pYPos)
{
    return findTransform(mOrientMatrices, pXPos, pYPos);
}

void Widget::setTransform(MatrixHashMap& pMap, const float pX, const float pY, const glm::mat4 &pMat)
{
    for (auto it: pMap) {
        const CellIndex& idx = it.first;

        const int rows = std::get<0>(idx);
        const int cols = std::get<1>(idx);

        const int cellWidth  = mWidth/cols;
        const int cellHeight = mHeight/rows;

        const int x = int(pX) / cellWidth;
        const int y = int(pY) / cellHeight;
        const int i = x + y * cols;
        if (i==std::get<2>(idx)) {
            pMap[idx] = pMat;
        }
    }
}

void Widget::setCellViewMatrix(const float pXPos, const float pYPos, const glm::mat4& pMatrix)
{
    return setTransform(mViewMatrices, pXPos, pYPos, pMatrix);
}

void Widget::setCellOrientationMatrix(const float pXPos, const float pYPos, const glm::mat4& pMatrix)
{
    return setTransform(mOrientMatrices, pXPos, pYPos, pMatrix);
}


void Widget::resetViewMatrices()
{
    for (auto it: mViewMatrices)
        it.second = IDENTITY;
}


void Widget::resetOrientationMatrices()
{
    for (auto it: mOrientMatrices)
        it.second = IDENTITY;
}

Widget::Widget()
    : mWindow(nullptr), mClose(false), mLastXPos(0), mLastYPos(0), mButton(-1),
    mWidth(512), mHeight(512), mFramePBO(0)
{
    initWindow(512, 512, "", nullptr);
}

Widget::Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible)
    : mWindow(nullptr), mClose(false), mLastXPos(0), mLastYPos(0), mButton(-1), mFramePBO(0)
{
    initWindow(pWidth, pHeight, pTitle, pWindow);
}

void Widget::initWindow(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow)
{
    //need to wait till glbinding has been init'ed before calling any gl functions
    //this will have been doen once resizePixelBuffers has been called
    mWindowCreated=false;

    if(qtApplication==nullptr)
    {
        setupApplication();
        mDockWidget=false;
    }
    else//main widow allready exist, make remaining windows docakable
        mDockWidget=true;

    if(pWindow!=nullptr)
    {
        QGLWidget *sharedWidget=dynamic_cast<QGLWidget *>(pWindow->getNativeHandle());

        if(sharedWidget!=nullptr)
            mWindow=new QGLWidget(nullptr, sharedWidget);
    }

    if(mWindow==nullptr)
        mWindow=new QGLWidget();

    mEventFilter=std::unique_ptr<EventFilter>(new EventFilter(this));
    mWindow->installEventFilter(mEventFilter.get());

    if(mDockWidget)
    {
        DockWrapper *wrapper=new DockWrapper(mWindow);

        qtMainWindow->addDockWidget(Qt::RightDockWidgetArea, wrapper);
        wrapper->setFloating(true);
    }
    else
        qtMainWindow->setCentralWidget(mWindow);

    if(mDockWidget)
        mWindow->resize(QSize(pWidth, pHeight));
    else
        qtMainWindow->resize(QSize(pWidth, pHeight));
    mWindow->show();

    mWidth=pWidth;
    mHeight=pHeight;

    qtMainWindow->show();

    qtApplication->processEvents();
}

Widget::~Widget()
{
    mWindow->removeEventFilter(mEventFilter.get());

    if(!mDockWidget)
        qtMainWindow->close();
}

QWidget *Widget::getNativeHandle() const
{
    return mWindow;
}

void Widget::makeContextCurrent() const
{
    mWindow->makeCurrent();
}

long long Widget::getGLContextHandle()
{
    return opengl::getCurrentContextHandle();
}

long long Widget::getDisplayHandle()
{
    return opengl::getCurrentDisplayHandle();
}

void Widget::setTitle(const char* pTitle)
{
    if(mDockWidget)
        mWindow->setWindowTitle(QString(pTitle));
    else
        qtMainWindow->setWindowTitle(QString(pTitle));
//    SDL_SetWindowTitle(mWindow, (pTitle!=nullptr ? pTitle : "Forge-Demo"));
}

void Widget::setPos(int pX, int pY)
{
    if(mDockWidget)
        mWindow->move(pX, pY);
    else
        qtMainWindow->move(pX, pY);
}

void Widget::setSize(unsigned pW, unsigned pH)
{
    if(mDockWidget)
        mWindow->resize(pW, pH);
    else
        qtMainWindow->resize(pW, pH);

    mWidth=pW;
    mHeight=pH;
}

void Widget::swapBuffers()
{
    mWindow->swapBuffers();

    glReadBuffer(GL_FRONT);
    glBindBuffer((gl::GLenum)GL_PIXEL_PACK_BUFFER, mFramePBO);
    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer((gl::GLenum)GL_PIXEL_PACK_BUFFER, 0);
}

void Widget::hide()
{
    if(mDockWidget)
        mWindow->hide();
    else
        qtMainWindow->hide();
}

void Widget::show()
{
    if(mDockWidget)
        mWindow->show();
    else
        qtMainWindow->show();
}

bool Widget::close()
{
    return !mWindow->isVisible();
}

void Widget::resetCloseFlag()
{
}

void Widget::pollEvents()
{
    if(!mDockWidget)
        qtApplication->processEvents();
}

bool Widget::eventFilter(QObject *object, QEvent *event)
{
    if(object!=mWindow)
        return false;

    switch(event->type())
    {
    case QEvent::Resize:
        QResizeEvent *resizeEvent=dynamic_cast<QResizeEvent *>(event);

        if(resizeEvent!=nullptr)
        {
            mWidth=resizeEvent->size().width();
            mHeight=resizeEvent->size().height();
            if(mWindowCreated)
                resizePixelBuffers();
        }
        break;
    }
    return false;
}

void Widget::resizePixelBuffers()
{
    mWindowCreated=true;

    if (mFramePBO!=0)
        glDeleteBuffers(1, &mFramePBO);

    uint w = mWidth;
    uint h = mHeight;

    glGenBuffers(1, &mFramePBO);
    glBindBuffer((gl::GLenum)GL_PIXEL_PACK_BUFFER, mFramePBO);
    glBufferData((gl::GLenum)GL_PIXEL_PACK_BUFFER, w*h*4*sizeof(uchar), 0, (gl::GLenum)GL_DYNAMIC_READ);
    glBindBuffer((gl::GLenum)GL_PIXEL_PACK_BUFFER, 0);
}

const glm::mat4 Widget::getViewMatrix(const CellIndex& pIndex)
{
    if (mViewMatrices.find(pIndex)==mViewMatrices.end()) {
        mViewMatrices.emplace(pIndex, IDENTITY);
    }
    return mViewMatrices[pIndex];
}

const glm::mat4 Widget::getOrientationMatrix(const CellIndex& pIndex)
{
    if (mOrientMatrices.find(pIndex)==mOrientMatrices.end()) {
        mOrientMatrices.emplace(pIndex, IDENTITY);
    }
    return mOrientMatrices[pIndex];
}

}
}
