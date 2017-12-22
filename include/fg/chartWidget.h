#ifndef _QChartWidget_h_
#define _QChartWidget_h_

#ifndef Q_MOC_RUN
#endif //Q_MOC_RUN

#include <QtOpenGL/QGLWidget>
#include <QtGui/QImage>

#include <fg/defines.h>
#include <fg/chart.h>

#include <atomic>
#include <thread>
#include <condition_variable>

namespace forge
{

class FGAPI ChartWidget: public QGLWidget
{
    Q_OBJECT

public:
    ChartWidget(forge::ChartType chartType, QWidget *parent=nullptr, bool threaded=false);
    ChartWidget(forge::ChartType chartType, QGLContext *context, QWidget *parent=nullptr, bool threaded=false);
    ~ChartWidget();

    /**
    Set axes titles for the chart

    \param[in] pX is x-axis title label
    \param[in] pY is y-axis title label
    \param[in] pZ is z-axis title label
    */
    void setAxesTitles(const char* pX,
        const char* pY,
        const char* pZ=NULL);

    /**
    Set axes data ranges

    \param[in] pXmin is x-axis minimum data value
    \param[in] pXmax is x-axis maximum data value
    \param[in] pYmin is y-axis minimum data value
    \param[in] pYmax is y-axis maximum data value
    \param[in] pZmin is z-axis minimum data value
    \param[in] pZmax is z-axis maximum data value
    */
    void setAxesLimits(const float pXmin, const float pXmax,
        const float pYmin, const float pYmax,
        const float pZmin=0, const float pZmax=0);

    /**
    Set the format for display of axes labels

    \param[in] pXFormat sets the display format for numbers of X axis
    \param[in] pYFormat sets the display format for numbers of Y axis
    \param[in] pZFormat sets the display format for numbers of Z axis

    Display format string follows printf style formating for numbers
    */
    void setAxesLabelFormat(const char* pXFormat,
        const char* pYFormat="%4.1f",
        const char* pZFormat="%4.1f");

    /**
    Get axes data ranges

    \param[out] pXmin is x-axis minimum data value
    \param[out] pXmax is x-axis maximum data value
    \param[out] pYmin is y-axis minimum data value
    \param[out] pYmax is y-axis maximum data value
    \param[out] pZmin is z-axis minimum data value
    \param[out] pZmax is z-axis maximum data value
    */
    void getAxesLimits(float* pXmin, float* pXmax,
        float* pYmin, float* pYmax,
        float* pZmin=NULL, float* pZmax=NULL);

    /**
    Set legend position for Chart

    \param[in] pX is horizontal position in normalized coordinates
    \param[in] pY is vertical position in normalized coordinates

    \note By normalized coordinates, the range of these coordinates is expected to be [0-1].
    (0,0) is the bottom hand left corner.
    */
    void setLegendPosition(const float pX, const float pY);

    /**
    Add an existing Image object to the current chart

    \param[in] pImage is the Image to render on the chart
    */
    void add(const Image& pImage);

    /**
    Add an existing Histogram object to the current chart

    \param[in] pHistogram is the Histogram to render on the chart
    */
    void add(const Histogram& pHistogram);

    /**
    Add an existing Plot object to the current chart

    \param[in] pPlot is the Plot to render on the chart
    */
    void add(const Plot& pPlot);

    /**
    Add an existing Surface object to the current chart

    \param[in] pSurface is the Surface to render on the chart
    */
    void add(const Surface& pSurface);

    /**
    Add an existing vector field object to the current chart

    \param[in] pVectorField is the Surface to render on the chart
    */
    void add(const VectorField& pVectorField);

    /**
    Remove an existing Image object from the current chart

    \param[in] pImage is the Image to remove from the chart
    */
    void remove(const Image& pImage);

    /**
    Remove an existing Histogram object from the current chart

    \param[in] pHistogram is the Histogram to remove from the chart
    */
    void remove(const Histogram& pHistogram);

    /**
    Remove an existing Plot object from the current chart

    \param[in] pPlot is the Plot to remove from the chart
    */
    void remove(const Plot& pPlot);

    /**
    Remove an existing Surface object from the current chart

    \param[in] pSurface is the Surface to remove from the chart
    */
    void remove(const Surface& pSurface);

    /**
    Remove an existing vector field object from the current chart

    \param[in] pVectorField is the Surface to remove from the chart
    */
    void remove(const VectorField& pVectorField);

    /**
    Create and add an Image object to the current chart

    \param[in] pWidth Width of the image
    \param[in] pHeight Height of the image
    \param[in] pFormat Color channel format of image, uses one of the values
    of \ref ChannelFormat
    \param[in] pDataType takes one of the values of \ref dtype that indicates
    the integral data type of histogram data
    */
    Image image(const unsigned pWidth, const unsigned pHeight,
        const ChannelFormat pFormat=FG_RGBA, const dtype pDataType=f32);

    /**
    Create and add an Histogram object to the current chart

    \param[in] pNBins is number of bins the data is sorted out
    \param[in] pDataType takes one of the values of \ref dtype that indicates
    the integral data type of histogram data
    */
    Histogram histogram(const unsigned pNBins, const dtype pDataType);

    /**
    Create and add an Plot object to the current chart

    \param[in] pNumPoints is number of data points to display
    \param[in] pDataType takes one of the values of \ref dtype that indicates
    the integral data type of plot data
    \param[in] pPlotType dictates the type of plot/graph,
    it can take one of the values of \ref PlotType
    \param[in] pMarkerType indicates which symbol is rendered as marker. It can take one of
    the values of \ref MarkerType.
    */
    Plot plot(const unsigned pNumPoints, const dtype pDataType,
        const PlotType pPlotType=FG_PLOT_LINE, const MarkerType pMarkerType=FG_MARKER_NONE);

    /**
    Create and add an Plot object to the current chart

    \param[in] pNumXPoints is number of data points along X dimension
    \param[in] pNumYPoints is number of data points along Y dimension
    \param[in] pDataType takes one of the values of \ref dtype that indicates
    the integral data type of plot data
    \param[in] pPlotType is the render type which can be one of \ref PlotType (valid choices
    are FG_PLOT_SURFACE and FG_PLOT_SCATTER)
    \param[in] pMarkerType is the type of \ref MarkerType to draw for \ref FG_PLOT_SCATTER plot type
    */
    Surface surface(const unsigned pNumXPoints, const unsigned pNumYPoints, const dtype pDataType,
        const PlotType pPlotType=FG_PLOT_SURFACE, const MarkerType pMarkerType=FG_MARKER_NONE);

    /**
    Create and add an Vector Field object to the current chart

    \param[in] pNumPoints is number of data points to display
    \param[in] pDataType takes one of the values of \ref dtype that indicates
    the integral data type of vector field data
    */
    VectorField vectorField(const unsigned pNumPoints, const dtype pDataType);

    /**
    Sets widgets opengl context
    */
    void setThreadContext(QGLContext *glContext);

    /**
    Intializes widget when used in threaded mode
    */
    void initialize();

    /**
    Breaks down widget when used in threaded mode
    */
    void terminate();

    /**
    Render the chart to the widget
    */
    void render();

    /**
    Notifies wait event
    */
    void update();

    /**
    Waits for update notification (resize, move, screen refresh) or timeout, return true it wait succeeded or false if timeout

    \param[in] timeout time in milliseconds to wait for update event before timingout
    */
    bool waitEvent(unsigned int timeout);

protected:
    bool event(QEvent *e);

    virtual void glInit();
    virtual void glDraw();

    virtual void initializeGL();
    virtual void resizeGL(int width, int height);
    virtual void paintGL();

    virtual void resizeEvent(QResizeEvent *evt);
    virtual void closeEvent(QCloseEvent *evt);



private:
    void resize();
    void renderInternal();

    void mousePressEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);

    int m_windowsId;
    forge::ChartType m_chartType;
    std::unique_ptr<forge::Chart> m_chart;
    bool m_threaded;

    std::atomic<bool> m_init;
    std::atomic<bool> m_resize;

    std::mutex m_mutex;
    std::condition_variable m_waitEvent;

    QPoint m_leftClick;
    QPoint m_rightClick;
};

}//namespace forge

#endif //_QChartWidget_h_
