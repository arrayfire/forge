#include "plot3MainWindow.h"

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

Plot3MainWindow::Plot3MainWindow(QWidget *parent, Qt::WindowFlags flags)
{
    m_chart=new forge::ChartWidget(FG_CHART_3D);
    
    m_chart->setAxesLabelFormat("%3.1f", "%3.1f", "%.2e");
    m_chart->setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, 0.f, 10.f);
    m_chart->setAxesTitles("x-axis", "y-axis", "z-axis");
    
    m_plot=std::unique_ptr<forge::Plot>(new forge::Plot(m_chart->plot(ZSIZE, forge::f32)));
    
    createGLBuffer(&m_functionHandle, m_plot->vertices(), FORGE_VERTEX_BUFFER);
    
    m_time=0.0f;
    generateCurve(m_time, DX, m_function);
    copyToGLBuffer(m_functionHandle, (ComputeResourceHandle)m_function.data(), m_plot->verticesSize());
    
    setCentralWidget(m_chart);
            
    m_drawTimer=new QTimer(this);
    
    bool connected=connect(m_drawTimer, SIGNAL(timeout()), this, SLOT(draw()));
    m_drawTimer->start(0);

    resize(400,400);
}

Plot3MainWindow::~Plot3MainWindow()
{
    m_drawTimer->stop();
    releaseGLBuffer(m_functionHandle);
}

void Plot3MainWindow::draw()
{
    m_time+=0.01f;
    generateCurve(m_time, DX, m_function);
    copyToGLBuffer(m_functionHandle, (ComputeResourceHandle)m_function.data(), m_plot->verticesSize());
    m_chart->render();
}
