#include "dockWrapper.hpp"

#ifndef Q_MOC_RUN 
#endif

namespace forge
{
namespace wtk
{

DockWrapper::DockWrapper(QWidget *widget, QWidget *parent):
    QDockWidget(parent)
{
    connect(widget, SIGNAL(accepted()), this, SLOT(childAccepted()));
    connect(widget, SIGNAL(rejected()), this, SLOT(childRejected()));
    connect(widget, SIGNAL(destroyed(QObject*)), this, SLOT(childDestroyed(QObject*)));

    setMinimumSize(widget->minimumSize());
    setMaximumSize(widget->maximumSize());
    setSizePolicy(widget->sizePolicy());

    setAttribute(Qt::WA_DeleteOnClose);
    setWidget(widget);
}

DockWrapper::~DockWrapper()
{

}

QSize DockWrapper::sizeHint() const
{
    return widget()->sizeHint();
}

QSize DockWrapper::minimumSizeHint() const
{
    return widget()->minimumSizeHint();
}

void DockWrapper::childAccepted()
{
    emit accepted();
}

void DockWrapper::childRejected()
{
    emit rejected();
}

void DockWrapper::childDestroyed(QObject *object)
{
    close();
}

}
}
