#include "fg/chart.h"
#include "handle.hpp"

#include "fg/chartWidget.h"

#include <QtGui/QImage>
#include <QtGui/QMouseEvent>
#include <QtCore/QThread>
#include <QtGui/QOpenGlContext>
#include <QtPlatformHeaders/QWGLNativeContext>
#include <QtGui/QWindow>

#include <gl_native_handles.hpp>

using namespace std::chrono_literals;

namespace forge
{
extern int getNextUniqueId();

ChartWidget::ChartWidget(forge::ChartType chartType, QWidget *parent, bool threaded):
QGLWidget(parent),
m_init(false),
m_threaded(threaded),
m_windowsId(getNextUniqueId()),
m_chartType(chartType)
{
    if(!threaded)
    {
        makeCurrent();
        initialize();
    }
    else
        doneCurrent();
}

ChartWidget::ChartWidget(forge::ChartType chartType, QGLContext *context, QWidget *parent, bool threaded):
QGLWidget(context, parent),
m_init(false),
m_threaded(threaded),
m_windowsId(getNextUniqueId()),
m_chartType(chartType)
{
    if(!threaded)
    {
        makeCurrent();
        initialize();
    }
    else
        doneCurrent();
}

ChartWidget::~ChartWidget()
{
    //if you hit this then you created the widget with threading
    //but didn't call terminate() before it was destroyed
    if(m_threaded && m_init)
        FG_ERROR("Chart has been created as threaded but terminate() has not been called before it was destroyed", FG_ERR_INTERNAL);
}
void ChartWidget::setAxesTitles(const char* pX, const char* pY, const char* pZ)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->setAxesTitles(pX, pY, pZ);
}

void ChartWidget::setAxesLimits(const float pXmin, const float pXmax,
    const float pYmin, const float pYmax,
    const float pZmin, const float pZmax)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->setAxesLimits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax);
}

void ChartWidget::setAxesLabelFormat(const char* pXFormat,
    const char* pYFormat,
    const char* pZFormat)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->setAxesLabelFormat(pXFormat, pYFormat, pZFormat);
}

void ChartWidget::getAxesLimits(float* pXmin, float* pXmax,
    float* pYmin, float* pYmax,
    float* pZmin, float* pZmax)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->getAxesLimits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax);
}

void ChartWidget::setLegendPosition(const float pX, const float pY)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->setLegendPosition(pX, pY);
}

void ChartWidget::add(const Image& pImage)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);
    
    m_chart->add(pImage);
}

void ChartWidget::add(const Histogram& pHistogram)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->add(pHistogram);
}

void ChartWidget::add(const Plot& pPlot)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->add(pPlot);
}

void ChartWidget::add(const Surface& pSurface)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->add(pSurface);
}

void ChartWidget::add(const VectorField& pVectorField)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->add(pVectorField);
}

void ChartWidget::remove(const Image& pImage)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->remove(pImage);
}

void ChartWidget::remove(const Histogram& pHistogram)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->remove(pHistogram);
}

void ChartWidget::remove(const Plot& pPlot)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->remove(pPlot);
}

void ChartWidget::remove(const Surface& pSurface)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->remove(pSurface);
}

void ChartWidget::remove(const VectorField& pVectorField)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    m_chart->remove(pVectorField);
}

Image ChartWidget::image(const unsigned pWidth, const unsigned pHeight,
    const ChannelFormat pFormat, const dtype pDataType)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    return m_chart->image(pWidth, pHeight, pFormat, pDataType);
}

Histogram ChartWidget::histogram(const unsigned pNBins, const dtype pDataType)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    return m_chart->histogram(pNBins, pDataType);
}

Plot ChartWidget::plot(const unsigned pNumPoints, const dtype pDataType,
    const PlotType pPlotType, const MarkerType pMarkerType)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    return m_chart->plot(pNumPoints, pDataType, pPlotType, pMarkerType);
}

Surface ChartWidget::surface(const unsigned pNumXPoints, const unsigned pNumYPoints, const dtype pDataType,
    const PlotType pPlotType, const MarkerType pMarkerType)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    return m_chart->surface(pNumXPoints, pNumYPoints, pDataType, pPlotType, pMarkerType);
}

VectorField ChartWidget::vectorField(const unsigned pNumPoints, const dtype pDataType)
{
    if(!m_init)
        FG_ERROR("Chart has not been initialized, if using threads you must call initialize or draw from thread before using", FG_ERR_INTERNAL);

    return m_chart->vectorField(pNumPoints, pDataType);
}

void ChartWidget::update()
{
    m_waitEvent.notify_all();
}

bool ChartWidget::waitEvent(unsigned int timeout)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    if(m_waitEvent.wait_for(lock, timeout*1ms)==std::cv_status::timeout)
        return false;
    return true;
}

void ChartWidget::glInit()
{
}

void ChartWidget::glDraw()
{
    update();
}

void ChartWidget::initializeGL()
{
}

void ChartWidget::resizeGL(int width, int height)
{
}

void ChartWidget::paintGL()
{
}

void ChartWidget::resizeEvent(QResizeEvent *evt)
{
    m_resize=true;
    update();
}

void ChartWidget::closeEvent(QCloseEvent *evt)
{
    update();
    QGLWidget::closeEvent(evt);
}

bool ChartWidget::event(QEvent *e)
{
    QEvent::Type type=e->type();

    if(e->type() == QEvent::Show)
        update();
    else if(e->type() == QEvent::ParentChange) //The glContext will be changed, need to reinit openGl
    {
        bool ret=QGLWidget::event(e);
        
        return ret;
    }
    else if(e->type() == QEvent::Resize)
    {
        return QGLWidget::event(e);
    }
    return QGLWidget::event(e);
}

void ChartWidget::setThreadContext(QGLContext *glContext)
{
    if(!m_threaded)
        FG_ERROR("Chart is not setup for threading, this function should not be called", FG_ERR_INTERNAL);

    setContext(glContext, nullptr, false);
    makeCurrent();
    initialize();
}

void ChartWidget::initialize()
{
    m_resize=true;
    setAutoBufferSwap(false);

    glbinding::Binding::useCurrentContext();
    glbinding::Binding::initialize();

    m_chart=std::unique_ptr<forge::Chart>(new forge::Chart(m_chartType));

    m_init=true;
}

void ChartWidget::terminate()
{
    m_chart.reset(nullptr);
    m_init=false;
}


void ChartWidget::resize()
{
    m_resize=false;
}


void ChartWidget::render()
{
    if(!isVisible())
        return;

    if(!windowHandle()->isExposed())
        return;

    if(!m_init)
        initialize();

    if(m_resize)
        resize();

    makeCurrent();

    glViewport(0, 0, width(), height());

    // clear color and depth buffers
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    //draw
    getChart(m_chart->get())->render(m_windowsId, 0, 0, width(), height(), IDENTITY, IDENTITY);

    swapBuffers();
}

void ChartWidget::mousePressEvent(QMouseEvent* event)
{
    Qt::MouseButton button=event->button();

    if(Qt::LeftButton == button)
        m_leftClick=event->globalPos();
    if(Qt::RightButton == button)
        m_rightClick=event->globalPos();
}

void ChartWidget::mouseReleaseEvent(QMouseEvent* event)
{
    this->setCursor(QCursor(Qt::OpenHandCursor));
}

void ChartWidget::mouseMoveEvent(QMouseEvent* event)
{
    Qt::MouseButtons buttons=event->buttons();

    if(Qt::LeftButton&buttons)
    {
        bool rightButton=Qt::RightButton&buttons;

        if(!rightButton)
        {
            Qt::KeyboardModifiers keys=event->modifiers();

            if(Qt::AltModifier&keys)
            {
            }
            else
            {
            }
        }
        m_leftClick = event->globalPos();
    }
}

}//namespace forge