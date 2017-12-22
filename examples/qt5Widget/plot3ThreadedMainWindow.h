#ifndef _Plot3ThreadedMainWindow_h_
#define _Plot3ThreadedMainWindow_h_

#include <QtWidgets/QMainWindow>
#include <QtCore/QThread>

#include <forge.h>

#define USE_FORGE_CPU_COPY_HELPERS
#include <ComputeCopy.h>

#include <thread>

class Plot3ThreadedMainWindow: public QMainWindow
{
    Q_OBJECT

public:
    Plot3ThreadedMainWindow(QWidget *parent=0, Qt::WindowFlags flags=0);
    ~Plot3ThreadedMainWindow();

public slots:
    void draw();

private:
    forge::ChartWidget *m_chart;
    
    std::atomic<bool> m_stopThread;
    std::thread m_drawThread;

    QThread *m_qtDrawThread;
    std::condition_variable m_qtDrawThreadEvent;
    std::condition_variable m_qtDrawThreadMoved;
    std::mutex m_qtDrawThreadMutex;

    bool m_contextMoved;
};

#endif //_Plot3ThreadedMainWindow_h_
