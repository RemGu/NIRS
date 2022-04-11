#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <vector>


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow), nPixSrc(0), nPixF(0), nPixDst(0)



{
    ui->setupUi(this);
    ui->srcGraphicsView->setScene(&scSrc);
    ui->fGraphicsView->setScene(&scF);
    ui->dstGraphicsView->setScene(&scDst);
    redMult = 0;
    greenMult = 0;


}

MainWindow::~MainWindow()
{
    delete ui;
}



QImage MainWindow::Mat2QImage(cv::Mat const& src)
{
    switch(src.type())
    {
    case CV_8UC4:
    {
        cv::Mat view(src);
        QImage view2(view.data, view.cols, view.rows, view.step[0], QImage::Format_ARGB32);
        return view2.copy();
        break;
    }
    case CV_8UC3:
    {
        cv::Mat mat;
        cvtColor(src, mat, cv::COLOR_BGR2BGRA); //COLOR_BGR2RGB doesn't behave so use RGBA
        QImage view(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_ARGB32);
        return view.copy();
        break;
    }
    case CV_8UC1:
    {
        cv::Mat mat;
        cvtColor(src, mat, cv::COLOR_GRAY2BGRA);
        QImage view(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_ARGB32);
        return view.copy();
        break;
    }
    default:
    {
        //throw invalid_argument("Image format not supported");
        return QImage();
        break;
    }
    }
}


cv::Mat MainWindow::QImage2Mat(QImage const& f)
{
    switch(f.format()) {
    case QImage::Format_Invalid:
    {
        cv::Mat empty ;
        return empty  ;
        break;
    }
    case QImage::Format_RGB32:
    {
        cv::Mat view(f.height(),f.width(),CV_8UC4,(void *)f.constBits(),f.bytesPerLine()) ;
        return view ;
        break;
    }
    case QImage::Format_RGB888:
    {
        cv::Mat out ;
        cv::Mat view(f.height(),f.width(),CV_8UC3,(void *)f.constBits(),f.bytesPerLine());
        cv::cvtColor(view, out, cv::COLOR_RGB2BGR);
        return out ;
        break;
    }
    default:
    {
        QImage conv = f.convertToFormat(QImage::Format_ARGB32);
        cv::Mat view(conv.height(),conv.width(),CV_8UC4,(void *)conv.constBits(),conv.bytesPerLine());
        return view ;
        break;
    }
    }
}

enum NiblackVersion
{
    NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
};

#define BINARIZEWOLF_VERSION	"2.4 (August 1st, 2014)"

#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;





double calcLocalStats (Mat &im, Mat &map_m, Mat &map_s, int winx, int winy) {
    Mat im_sum, im_sum_sq;
    cv::integral(im,im_sum,im_sum_sq,CV_64F);

    double m,s,max_s,sum,sum_sq;
    int wxh	= winx/2;
    int wyh	= winy/2;
    int x_firstth= wxh;
    int y_firstth= wyh;
    int y_lastth = im.rows-wyh-1;
    double winarea = winx*winy;

    max_s = 0;
    for	(int j = y_firstth ; j<=y_lastth; j++){
        sum = sum_sq = 0;

        // for sum array iterator pointer
        double *sum_top_left = im_sum.ptr<double>(j - wyh);
        double *sum_top_right = sum_top_left + winx;
        double *sum_bottom_left = im_sum.ptr<double>(j - wyh + winy);
        double *sum_bottom_right = sum_bottom_left + winx;

        // for sum_sq array iterator pointer
        double *sum_eq_top_left = im_sum_sq.ptr<double>(j - wyh);
        double *sum_eq_top_right = sum_eq_top_left + winx;
        double *sum_eq_bottom_left = im_sum_sq.ptr<double>(j - wyh + winy);
        double *sum_eq_bottom_right = sum_eq_bottom_left + winx;

        sum = (*sum_bottom_right + *sum_top_left) - (*sum_top_right + *sum_bottom_left);
        sum_sq = (*sum_eq_bottom_right + *sum_eq_top_left) - (*sum_eq_top_right + *sum_eq_bottom_left);

        m  = sum / winarea;
        s  = sqrt ((sum_sq - m*sum)/winarea);
        if (s > max_s) max_s = s;

        float *map_m_data = map_m.ptr<float>(j) + x_firstth;
        float *map_s_data = map_s.ptr<float>(j) + x_firstth;
        *map_m_data++ = m;
        *map_s_data++ = s;

        // Shift the window, add and remove	new/old values to the histogram
        for	(int i=1 ; i <= im.cols-winx; i++) {
            sum_top_left++, sum_top_right++, sum_bottom_left++, sum_bottom_right++;

            sum_eq_top_left++, sum_eq_top_right++, sum_eq_bottom_left++, sum_eq_bottom_right++;

            sum = (*sum_bottom_right + *sum_top_left) - (*sum_top_right + *sum_bottom_left);
            sum_sq = (*sum_eq_bottom_right + *sum_eq_top_left) - (*sum_eq_top_right + *sum_eq_bottom_left);

            m  = sum / winarea;
            s  = sqrt ((sum_sq - m*sum)/winarea);
            if (s > max_s) max_s = s;

            *map_m_data++ = m;
            *map_s_data++ = s;
        }
    }

    return max_s;
}



void NiblackSauvolaWolfJolion (Mat im, Mat output, NiblackVersion version,
    int winx, int winy, double k, double dR) {


    double m, s, max_s;
    double th=0;
    double min_I, max_I;
    int wxh	= winx/2;
    int wyh	= winy/2;
    int x_firstth= wxh;
    int x_lastth = im.cols-wxh-1;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    // int mx, my;

    // Create local statistics and store them in a double matrices
    Mat map_m = Mat::zeros (im.rows, im.cols, CV_32F);
    Mat map_s = Mat::zeros (im.rows, im.cols, CV_32F);
    max_s = calcLocalStats (im, map_m, map_s, winx, winy);

    minMaxLoc(im, &min_I, &max_I);

    Mat thsurf (im.rows, im.cols, CV_32F);

    
    for	(int j = y_firstth ; j<=y_lastth; j++) {

        float *th_surf_data = thsurf.ptr<float>(j) + wxh;
        float *map_m_data = map_m.ptr<float>(j) + wxh;
        float *map_s_data = map_s.ptr<float>(j) + wxh;

        // NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
        for	(int i=0 ; i <= im.cols-winx; i++) {
            m = *map_m_data++;
            s = *map_s_data++;

            // Calculate the threshold
            switch (version) {

                case NIBLACK:
                    th = m + k*s;
                    break;

                case SAUVOLA:
                    th = m * (1 + k*(s/dR-1));
                    break;

                case WOLFJOLION:
                    th = m + k * (s/max_s-1) * (m-min_I);
                    break;

                default:

                    exit (1);
            }

            // thsurf.fset(i+wxh,j,th);
            *th_surf_data++ = th;


            if (i==0) {
                // LEFT BORDER
                float *th_surf_ptr = thsurf.ptr<float>(j);
                for (int i=0; i<=x_firstth; ++i)
                    *th_surf_ptr++ = th;

                // LEFT-UPPER CORNER
                if (j==y_firstth)
                {
                    for (int u=0; u<y_firstth; ++u)
                    {
                        float *th_surf_ptr = thsurf.ptr<float>(u);
                        for (int i=0; i<=x_firstth; ++i)
                            *th_surf_ptr++ = th;
                    }

                }

                // LEFT-LOWER CORNER
                if (j==y_lastth)
                {
                    for (int u=y_lastth+1; u<im.rows; ++u)
                    {
                        float *th_surf_ptr = thsurf.ptr<float>(u);
                        for (int i=0; i<=x_firstth; ++i)
                            *th_surf_ptr++ = th;
                    }
                }
            }

            // UPPER BORDER
            if (j==y_firstth)
                for (int u=0; u<y_firstth; ++u)
                    thsurf.fset(i+wxh,u,th);

            // LOWER BORDER
            if (j==y_lastth)
                for (int u=y_lastth+1; u<im.rows; ++u)
                    thsurf.fset(i+wxh,u,th);
        }

        // RIGHT BORDER
        float *th_surf_ptr = thsurf.ptr<float>(j) + x_lastth;
        for (int i=x_lastth; i<im.cols; ++i)
            // thsurf.fset(i,j,th);
            *th_surf_ptr++ = th;

        // RIGHT-UPPER CORNER
        if (j==y_firstth)
        {
            for (int u=0; u<y_firstth; ++u)
            {
                float *th_surf_ptr = thsurf.ptr<float>(u) + x_lastth;
                for (int i=x_lastth; i<im.cols; ++i)
                    *th_surf_ptr++ = th;
            }
        }

        // RIGHT-LOWER CORNER
        if (j==y_lastth)
        {
            for (int u=y_lastth+1; u<im.rows; ++u)
            {
                float *th_surf_ptr = thsurf.ptr<float>(u) + x_lastth;
                for (int i=x_lastth; i<im.cols; ++i)
                    *th_surf_ptr++ = th;
            }
        }
    }


    for	(int y=0; y<im.rows; ++y)
    {
        unsigned char *im_data = im.ptr<unsigned char>(y);
        float *th_surf_data = thsurf.ptr<float>(y);
        unsigned char *output_data = output.ptr<unsigned char>(y);
        for	(int x=0; x<im.cols; ++x)
        {
            *output_data = *im_data >= *th_surf_data ? 255 : 0;
            im_data++;
            th_surf_data++;
            output_data++;
        }
    }
}


void MainWindow::GrayPicture()
{
    if(nPixSrc != nullptr)
    {
    QImage image = nPixSrc->pixmap().toImage() ;
    int height = image.height();
    int width = image.width();
    int hist[256];
    for(int i = 0; i < 256; i++) hist[i] = 0;

    for(int y = 0; y < height; ++y )
    {
        for( int x = 0; x < width; ++x)
        {
            int r = qRed(image.pixel(x,y));
            int g = qGreen(image.pixel(x,y));
            int b = qBlue(image.pixel(x,y));
            float blueMult = 1 - redMult - greenMult;
            int i = (int)( redMult * (float)r + greenMult * (float)g + blueMult * (float)b);
            hist[i]++;
            image.setPixel( x, y, qRgb(i, i, i));
        }
    }




    if(nPixDst == nullptr)
        nPixDst = new nGPixmapItem(QPixmap::fromImage(image));
    else
        nPixDst->setPixmap(QPixmap::fromImage(image));

    scDst.addItem(nPixDst);
    }
}

int otsuThreshold(int* hist, int size)
{
  // Введем два вспомогательных числа:
  int m = 0; // m - сумма высот всех бинов, домноженных на положение их середины
  int n = 0; // n - сумма высот всех бинов
  for (int i = 0; i < size; ++i)
  {
    m += i * hist[i];
    n += hist[i];
  }

  float maxSigma = -1; // Максимальное значение межклассовой дисперсии
  int threshold = 0; // Порог, соответствующий maxSigma

  int alpha1 = 0; // Сумма высот всех бинов для класса 1
  int beta1 = 0; // Сумма высот всех бинов для класса 1, домноженных на положение их середины

  for (int i = 0; i < size; ++i)
  {
    alpha1 += i * hist[i];
    beta1 += hist[i];

    float w1 = (float)beta1 / n;
    float a = (float)alpha1 / beta1 - (float)(m - alpha1) / (n - beta1);
    float sigma = w1 * (1 - w1) * a * a;

    if (sigma > maxSigma)
    {
      maxSigma = sigma;
      threshold = i;
    }
  }

  return threshold;
}

void MainWindow::on_openButton_clicked()
{

    QStringList fp = QFileDialog::getOpenFileNames();
    fd = QFileInfo(fp[0]).absolutePath();

    for( int i = 0 ; i < fp.count() ; ++i )
    {
        ui->listWidget->addItem(QFileInfo(fp[i]).fileName());
    }
    
}

void MainWindow::on_clearButton_clicked()
{
    scSrc.clear() ;
    scDst.clear() ;
    scF.clear() ;
}

void MainWindow::on_saveButton_clicked()
{
    if (nPixDst != nullptr)
            nPixDst->pixmap().toImage().save(QFileDialog::getSaveFileName(), "JPG");
}

void MainWindow::on_pushButton_3_clicked()
{
    ui->srcGraphicsView->scale(1/1.1, 1/1.1);
    ui->fGraphicsView->scale(1/1.1, 1/1.1);
    ui->dstGraphicsView->scale(1/1.1, 1/1.1);

}

void MainWindow::on_bigButton_clicked()
{
    ui->srcGraphicsView->scale(1.1, 1.1);
    ui->dstGraphicsView->scale(1.1, 1.1);
    ui->fGraphicsView->scale(1.1, 1.1);
}

void MainWindow::on_listWidget_itemClicked(QListWidgetItem *item)
{
    if( !(nPixSrc || nPixDst || nPixF ) )
    {
        delete nPixSrc;
        delete nPixDst;
         delete nPixF;

    }

    nPixSrc = new nGPixmapItem(QPixmap(QString(fd)+ "/" + item->text()));
    nPixDst = new nGPixmapItem(QPixmap(QString(fd)+ "/" + item->text()));
    nPixF = new nGPixmapItem(QPixmap(QString(fd)+ "/" + item->text()));




    scSrc.addItem(nPixSrc); //add to scene
    scDst.addItem(nPixDst);
    scF.addItem(nPixF);
}




void MainWindow::on_conButton_clicked()
{
   

    //GrayPicture();
    QImage im = nPixSrc->pixmap().toImage();
    cv::Mat bgr_image = QImage2Mat(im);
       cv::Mat lab_image;
       cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

       // Extract the L channel
       std::vector<cv::Mat> lab_planes(3);
       cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

       // apply the CLAHE algorithm to the L channel
       cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
       clahe->setClipLimit(1);
       cv::Mat dst;
       clahe->apply(lab_planes[0], dst);

       // Merge the the color planes back into an Lab image
       dst.copyTo(lab_planes[0]);
       cv::merge(lab_planes, lab_image);

      // convert back to RGB
      cv::Mat image_clahe;
      cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
   nPixF->setPixmap( QPixmap::fromImage( Mat2QImage( image_clahe)) );



  


}


void MainWindow::on_binButton_clicked()
{
  
    if(nPixF != nullptr)
    {
        QImage image = nPixF->pixmap().toImage();

        int hist[256];
        for (int i = 0; i < 256; ++i) hist[i] = 0;

        int width = image.width();
        int height = image.height();

        for(int y = 0; y < height; ++y )
        {
            for( int x = 0; x < width; ++x)
            {
                int r = qRed(image.pixel(x,y));
                int g = qGreen(image.pixel(x,y));
                int b = qBlue(image.pixel(x,y));
                float blueMult = 1 - redMult - greenMult;
                int i = (int)( redMult * (float)r + greenMult * (float)g + blueMult * (float)b);
                                    //if (i <= 254)
                hist[i]++;
            }
        }

        int threshold = otsuThreshold(hist, 256);

        for (int y = 0; y < height; ++y )
        {
            for (int x = 0; x < width; ++x)
            {
                int r = qRed(image.pixel(x,y));
                int g = qGreen(image.pixel(x,y));
                int b = qBlue(image.pixel(x,y));
                float blueMult = 1 - redMult - greenMult;
                int i = (int)( redMult * (float)r + greenMult * (float)g + blueMult * (float)b);
                if (i > threshold)
                    image.setPixel(x, y, qRgb(0, 0, 0));

            }
        }

        if(nPixDst == nullptr)
            nPixDst = new nGPixmapItem(QPixmap::fromImage(image));
        else
            nPixDst->setPixmap(QPixmap::fromImage(image));

        scDst.addItem(nPixDst);
    }



}



void MainWindow::on_pushButton_clicked()
{

    using namespace cv;

    QImage temp = nPixSrc->pixmap().toImage();

    Mat inpImg, outImg ;
    inpImg = QImage2Mat(temp);


    Mat kernel = (Mat_<float>(3,3) <<
                      1,  1, 1,
                      1, -9, 1,
                      1,  1, 1);


    Mat imgLaplacian;
    filter2D(inpImg, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    inpImg.convertTo(sharp, CV_32F);
    outImg = sharp - imgLaplacian;


    outImg.convertTo(outImg, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
        nPixDst->setPixmap( QPixmap::fromImage( Mat2QImage( outImg )) );
 }

void MainWindow::on_pushButton_2_clicked()
{

    GrayPicture();
}

void MainWindow::on_pushButton_4_clicked()
{
    int c;
        int winx=0, winy=0;
        float optK=0.5;
        bool didSpecifyK=false;
        NiblackVersion versionCode = NIBLACK;
        QImage img = nPixF->pixmap().toImage();
        Mat input, helper;
        input = QImage2Mat(img);
        cvtColor(input, helper, COLOR_RGB2GRAY);


            if ((helper.rows<=0) || (helper.cols<=0)) {

                exit(1);
            }


            // Treat the window size
            if (winx==0||winy==0) {

                winy = (int) (2.0 * helper.rows-1)/3;
                winx = (int) helper.cols-1 < winy ? helper.cols-1 : winy;
                // if the window is too big, than we asume that the image
                // is not a single text box, but a document page: set
                // the window size to a fixed constant.
                if (winx > 100)
                    winx = winy = 40;

            }

            // Threshold
            Mat output (helper.rows, helper.cols, CV_8U);
            NiblackSauvolaWolfJolion (helper, output, versionCode, winx, winy, optK, 128);

            // Write the tresholded file

           nPixDst->setPixmap( QPixmap::fromImage( Mat2QImage( output )) );

}

void MainWindow::on_pushButton_5_clicked()
{
    int c;
        int winx=0, winy=0;
        float optK=0.5;
        bool didSpecifyK=false;
        NiblackVersion versionCode = SAUVOLA;
        QImage img = nPixF->pixmap().toImage();
        Mat input, helper;
        input = QImage2Mat(img);
        cvtColor(input, helper, COLOR_RGB2GRAY);


            if ((helper.rows<=0) || (helper.cols<=0)) {

                exit(1);
            }


            // Treat the window size
            if (winx==0||winy==0) {

                winy = (int) (2.0 * helper.rows-1)/3;
                winx = (int) helper.cols-1 < winy ? helper.cols-1 : winy;
                // if the window is too big, than we asume that the image
                // is not a single text box, but a document page: set
                // the window size to a fixed constant.
                if (winx > 100)
                    winx = winy = 40;

            }

            // Threshold
            Mat output (helper.rows, helper.cols, CV_8U);
            NiblackSauvolaWolfJolion (helper, output, versionCode, winx, winy, optK, 128);

            // Write the tresholded file

           nPixDst->setPixmap( QPixmap::fromImage( Mat2QImage( output )) );

}

void MainWindow::on_pushButton_6_clicked()
{
    int c;
        int winx=0, winy=0;
        float optK=0.5;
        bool didSpecifyK=false;
        NiblackVersion versionCode = WOLFJOLION;
        QImage img = nPixF->pixmap().toImage();
        Mat input, helper;
        input = QImage2Mat(img);
        cvtColor(input, helper, COLOR_RGB2GRAY);


            if ((helper.rows<=0) || (helper.cols<=0)) {

                exit(1);
            }


            // Treat the window size
            if (winx==0||winy==0) {

                winy = (int) (2.0 * helper.rows-1)/3;
                winx = (int) helper.cols-1 < winy ? helper.cols-1 : winy;
                // if the window is too big, than we asume that the image
                // is not a single text box, but a document page: set
                // the window size to a fixed constant.
                if (winx > 100)
                    winx = winy = 40;

            }

            // Threshold
            Mat output (helper.rows, helper.cols, CV_8U);
            NiblackSauvolaWolfJolion (helper, output, versionCode, winx, winy, optK, 128);

            // Write the tresholded file

           nPixDst->setPixmap( QPixmap::fromImage( Mat2QImage( output )) );
}

void MainWindow::on_pushButton_7_clicked()
{
    using namespace cv;

    QImage temp = nPixF->pixmap().toImage();

    Mat inpImg, outImg, bw ;
    inpImg = QImage2Mat(temp);
    inpImg.convertTo(inpImg, CV_8UC3);
    cvtColor(inpImg, bw, CV_BGR2GRAY);


    adaptiveThreshold(bw, outImg, 255, ADAPTIVE_THRESH_MEAN_C,
             THRESH_BINARY, 11, 12);



        nPixDst->setPixmap( QPixmap::fromImage( Mat2QImage( outImg )) );
}
