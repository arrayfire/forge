/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
*
* Original implementation from LimitlessSDK
* https://github.com/InfiniteInteractive/LimitlessSDK/blob/master/sdk/QtComponents/dockWrapper.h
********************************************************/

#pragma once

#include <QtWidgets/QDockWidget>

namespace forge
{
namespace wtk
{

class DockWrapper: public QDockWidget
{
    Q_OBJECT

public:
    DockWrapper(QWidget *widget, QWidget *parent=0);
    ~DockWrapper();

    virtual QSize sizeHint() const;
    virtual QSize minimumSizeHint() const;

 public slots:
    void childAccepted();
    void childRejected();
    void childDestroyed(QObject *object);

signals:
    void accepted();
    void rejected();

protected:

private:

};

}
}
