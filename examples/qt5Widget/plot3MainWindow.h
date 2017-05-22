#ifndef _Plot3MainWindow_h_
#define _Plot3MainWindow_h_

#include <QtWidgets/QMainWindow>
#include <QtCore/QTimer>

#include <forge.h>

#define USE_FORGE_CPU_COPY_HELPERS
#include <ComputeCopy.h>

class Plot3MainWindow: public QMainWindow
{
    Q_OBJECT

public:
    Plot3MainWindow(QWidget *parent=0, Qt::WindowFlags flags=0);
    ~Plot3MainWindow();

public slots:
    void draw();

private:
    forge::ChartWidget *m_chart;
    std::unique_ptr<forge::Plot> m_plot;

    QTimer *m_drawTimer;

    GfxHandle *m_functionHandle;
    std::vector<float> m_function;
    float m_time;
};

#endif //_Plot3MainWindow_h_
