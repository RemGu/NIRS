#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include "ngpixmapitem.h"
#include "ngpixmapitem.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <QListWidgetItem>
#include <QImage>
#include <QImageReader>
#include<QObject>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>








#include <QFileDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    QImage Mat2QImage( cv::Mat const& src ) ;
    cv::Mat QImage2Mat( QImage const& src ) ;
    //connect(ui->srcGraphicsView->verticalScrollBar(), SIGNAL(sliderMoved(int)), ui->dstGraphicsView->verticalScrollBar(),  SLOT(setValue(int)));
   // connect(ui->dstGraphicsView->verticalScrollBar(), SIGNAL(sliderMoved(int)), ui->srcGraphicsView->verticalScrollBar(),SLOT(setValue(int)));

    //connect(ui->srcGraphicsView->horizontalScrollBar(), SIGNAL(sliderMoved(int)), ui->dstGraphicsView->horizontalScrollBar(), SLOT(setValue(int)));
    //connect(ui->dstGraphicsView->horizontalScrollBar(), SIGNAL(sliderMoved(int)), ui->srcGraphicsView->horizontalScrollBar(), SLOT(setValue(int)));



private slots:
    void on_openButton_clicked();

    void on_clearButton_clicked();

    void on_saveButton_clicked();

    void on_pushButton_3_clicked();

    void on_bigButton_clicked();

    void on_listWidget_itemClicked(QListWidgetItem *item);

    void on_conButton_clicked();

    void on_binButton_clicked();

    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_4_clicked();

    void on_pushButton_5_clicked();

    void on_pushButton_6_clicked();

    void on_pushButton_7_clicked();

private:
    Ui::MainWindow *ui;
     QGraphicsScene *scene;

     QGraphicsScene scSrc;
     QGraphicsScene scF;
     QGraphicsScene scDst;

     QString fd;
     nGPixmapItem *nPixSrc;
     nGPixmapItem *nPixF;
     nGPixmapItem *nPixDst;
     float redMult;
     float greenMult;
     void GrayPicture();
     QImage source;
     QImage help;

};
#endif // MAINWINDOW_H
