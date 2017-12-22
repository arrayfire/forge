#include "plot3ThreadedMainWindow.h"

#include <complex>
#include <cmath>
#include <vector>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float ZMIN = 0.1f;
static const float ZMAX = 10.f;

const float DX = 0.005f;
const size_t ZSIZE = (ZMAX-ZMIN)/DX+1;

using namespace std;

void generateCurve(float t, float dx, std::vector<float> &vec )
{
    vec.clear();
    
    for (int i=0; i < (int)ZSIZE; ++i)
    {
        float z = ZMIN + i*dx;
        vec.push_back(cos(z*t+t)/z);
        vec.push_back(sin(z*t+t)/z);
        vec.push_back(z+0.1*sin(t));
    }
}

Plot3ThreadedMainWindow::Plot3ThreadedMainWindow(QWidget *parent, Qt::WindowFlags flags):
m_qtDrawThread(nullptr)
{
    m_chart=new forge::ChartWidget(FG_CHART_3D, nullptr, true);
    setCentralWidget(m_chart);
            
    m_contextMoved=false;
    m_stopThread=false;
    m_drawThread=std::thread(std::bind(&Plot3ThreadedMainWindow::draw, this));

    //This could be done with a QThread but I prefer std::thread
    //so there is a bit of a dance to get them in sync because
    //only the ui thread can move the context and a QThread
    //cannot be created from a std::thread

    //You could also skip all this and create the opengl context in the
    //thread and use ChartWidget.setThreadContext()

    //if you are not familar with threads, condition_variable's wait 
    //releases the lock while waiting
    {
        //wait for thread to set qtDrawThread
        std::unique_lock<std::mutex> lock(m_qtDrawThreadMutex);

        while(m_qtDrawThread==nullptr)
            m_qtDrawThreadEvent.wait(lock);

        //now need to move the chart opengl context to the thread
        m_chart->context()->moveToThread(m_qtDrawThread);

        //now notify thread it is safe to proceed
        m_contextMoved=true;
    }
    m_qtDrawThreadMoved.notify_all();

    resize(400,400);
}

Plot3ThreadedMainWindow::~Plot3ThreadedMainWindow()
{
    m_stopThread=true;
    m_drawThread.join();
}

void Plot3ThreadedMainWindow::draw()
{
    {
        std::unique_lock<std::mutex> lock(m_qtDrawThreadMutex);

        //get qt thread, and notify ui thread we have it
        m_qtDrawThread=QThread::currentThread();
        m_qtDrawThreadEvent.notify_all();

        //wait for ui thread to move the context
        while(!m_contextMoved)
            m_qtDrawThreadMoved.wait(lock);
    }

    std::unique_ptr<forge::Plot> plot;
    GfxHandle *functionHandle;
    std::vector<float> function;
    float time;

    //make sure context is set for thread
    m_chart->makeCurrent();

    //intialize chart
    m_chart->initialize();

    m_chart->setAxesLabelFormat("%3.1f", "%3.1f", "%.2e");
    m_chart->setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, 0.f, 10.f);
    m_chart->setAxesTitles("x-axis", "y-axis", "z-axis");

    plot=std::unique_ptr<forge::Plot>(new forge::Plot(m_chart->plot(ZSIZE, forge::f32)));

    createGLBuffer(&functionHandle, plot->vertices(), FORGE_VERTEX_BUFFER);

    time=0.0f;
    while(!m_stopThread)
    {
        generateCurve(time, DX, function);
        copyToGLBuffer(functionHandle, (ComputeResourceHandle)function.data(), plot->verticesSize());
        m_chart->render();
        time+=0.01f;
    }

    //break down chart
    m_chart->terminate();

    releaseGLBuffer(functionHandle);
}
