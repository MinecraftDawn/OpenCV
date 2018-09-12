//
// Created by Eric on 2018/9/12.
//

#ifndef OPENCV_SHOWIMAGE_H
#define OPENCV_SHOWIMAGE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdint.h>

using namespace cv;
using namespace std;


class ShowImage {
public:

    void showImg() {
        char ig[] = "B:\\血小板.jpg";

        // 載入圖檔
        Mat image = imread(ig, CV_LOAD_IMAGE_COLOR);

        // 檢查讀檔是否成功
        if (!image.data) {
            cout << "無法開啟或找不到圖檔" << std::endl;
            return;
        }

        // 建立顯示圖檔視窗
        namedWindow("Display window", CV_WINDOW_NORMAL);

        // CV_WINDOW_FREERATIO 與 CV_WINDOW_KEEPRATIO
        // CV_GUI_NORMAL 與 CV_GUI_EXPANDED

        // 在視窗內顯示圖檔
        imshow("Display window", image);

        // 視窗等待按鍵
        waitKey(0);
    }

    void showImg2() {
        char ig[] = "B:\\血小板.jpg";

        // 載入圖檔
        Mat image = imread(ig, CV_LOAD_IMAGE_COLOR);

        imshow("2", image);
        waitKey(0);
    }

    void showImg3() { //灰階
        char ig[] = "B:\\血小板.jpg";

        Mat testColor = imread(ig, CV_LOAD_IMAGE_GRAYSCALE);

        namedWindow("Display window", CV_WINDOW_NORMAL);

        imshow("Display window", testColor);

        imwrite("B:\\血小板灰.jpg", testColor);

        waitKey(0);
    }

    void showImg4() {//視窗調整
        char ig[] = "B:\\血小板.jpg";

        Mat file = imread(ig, CV_LOAD_IMAGE_GRAYSCALE);

        namedWindow("window1", CV_WINDOW_NORMAL);
        namedWindow("window2", CV_WINDOW_AUTOSIZE);

        imshow("window1", file);
        imshow("window2", file);

        resizeWindow("window1", file.cols / 2, file.rows / 2);

        moveWindow("window1", -500, -500);

        waitKey(0);
    }

    void showImg5() {//RGB
        char ig[] = "B:\\血小板.jpg";

        Mat file = imread(ig, CV_LOAD_IMAGE_COLOR);
        Mat file2 = imread(ig, CV_LOAD_IMAGE_COLOR);

        for (int i = 0; i < file.rows; ++i) {
            for (int j = 0; j < file.cols; ++j) {
                file.at<Vec3b>(i, j)[0] = file.at<Vec3b>(i, j)[0] * 1;//B
                file.at<Vec3b>(i, j)[1] = file.at<Vec3b>(i, j)[1] * 1;//G
                file.at<Vec3b>(i, j)[2] = file.at<Vec3b>(i, j)[2] * 1;//R
            }
        }

        imshow("Display", file);
        imshow("D2", file2);
        waitKey(0);


    }

};


#endif //OPENCV_SHOWIMAGE_H
