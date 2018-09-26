//
// Created by Eric on 2018/9/12.
//

#ifndef OPENCV_SHOWIMAGE_H
#define OPENCV_SHOWIMAGE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv\cv.h>
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

    void showImg5() { //自訂灰階
        char ig[] = "B:\\血小板.jpg";

        Mat file = imread(ig, CV_LOAD_IMAGE_GRAYSCALE);

        for (int i = 0; i < file.rows; ++i) {
            for (int j = 0; j < file.cols; ++j) {
                file.at<uint8_t>(i, j) = file.at<uint8_t>(i, j) * 0.8;
            }
        }

        imshow("Display", file);
        waitKey(0);
    }

    void showImg6() {//RGB
        char ig[] = "B:\\血小板.jpg";

        Mat file = imread(ig, CV_LOAD_IMAGE_COLOR);
        Mat file2 = imread(ig, CV_LOAD_IMAGE_COLOR);

        for (int i = 0; i < file.rows; ++i) {
            for (int j = 0; j < file.cols; ++j) {
                file.at<Vec3b>(i, j)[0] = file.at<Vec3b>(i, j)[0] * 1;//B
                file.at<Vec3b>(i, j)[1] = file.at<Vec3b>(i, j)[1] * 1;//G
                file.at<Vec3b>(i, j)[2] = file.at<Vec3b>(i, j)[2] * 0;//R
            }
        }

        imshow("Display", file);
        imshow("D2", file2);
        waitKey(0);

    }

    void showImg7() { //merge
        char ig[] = "B:\\血小板.jpg";

        Mat file = imread(ig, CV_LOAD_IMAGE_COLOR);

        Mat splitC[3];

        split(file, splitC);

        imshow("B", splitC[0]);
        imshow("G", splitC[1]);
        imshow("R", splitC[2]);

        splitC[0] = Mat::zeros(splitC[0].size(), CV_8UC1);
        splitC[1] = Mat::zeros(splitC[1].size(), CV_8UC1);
        splitC[2] = Mat::zeros(splitC[2].size(), CV_8UC1);

        Mat output;

        merge(splitC, 3, output);

        imshow("Merged", output);

        waitKey(0);
    }

    void showImg8() {
        char ig[] = "B:\\血小板.jpg";

        Mat file = imread(ig, CV_LOAD_IMAGE_COLOR);

        Mat originalFloat;

        file.convertTo(originalFloat, CV_32FC2, 1.0 / 255.0);

        Mat dftOfOriginal;

        Mat originalComplex[2] = {originalFloat, Mat::zeros(originalFloat.size(), CV_32F)};

        Mat dftReady;

        merge(originalComplex, 2, dftReady);

        dft(dftReady, dftOfOriginal, DFT_COMPLEX_OUTPUT);

        Mat splitArray[2] = {Mat::zeros(dftOfOriginal.size(), CV_32F), Mat::zeros(dftOfOriginal.size(), CV_32F)};

        split(dftOfOriginal, splitArray);

        Mat dftMagnitude;

        magnitude(splitArray[0], splitArray[1], dftMagnitude);

        dftMagnitude += Scalar::all(1);

        log(dftMagnitude, dftMagnitude);

        normalize(dftMagnitude, dftMagnitude, 0, 1, CV_MINMAX);

        imshow("DFT", dftMagnitude);

        waitKey(0);

    }

    void showImg9() {
        char img[] = "B:\\血小板.jpg";

        Mat image;

        // 載入圖檔
        image = imread(img, CV_LOAD_IMAGE_COLOR);

        // 檢查讀檔是否成功
        if (!image.data) {
            cout << "無法開啟或找不到圖檔" << std::endl;
            return;
        }

        // 建立顯示圖檔視窗
        namedWindow("原圖", CV_WINDOW_NORMAL);
        namedWindow("下雪圖", CV_WINDOW_NORMAL);

        imshow("原圖", image);

        // 雪點數
        int i = 600;
        int color = 255;

        for (int k = 0; k < i; k++) {

            if (k == 300) {
                color = 0;
            }

            int i = rand() % image.cols;
            int j = rand() % image.rows;

            if (image.channels() == 1) { // gray-level image
                image.at<uchar>(j, i) = color;
                if (i < (int) image.cols)
                    image.at<uchar>(j + 1, i) = color;
                if (j < (int) image.rows)
                    image.at<uchar>(j, i + 1) = color;
                if (i < (int) image.cols && j < (int) image.rows)
                    image.at<uchar>(j + 1, i + 1) = color;

            } else if (image.channels() == 3) { // color image
                image.at<cv::Vec3b>(j, i)[0] = color;
                image.at<cv::Vec3b>(j, i)[1] = color;
                image.at<cv::Vec3b>(j, i)[2] = color;

                if (i < (int) image.cols - 1) {
                    image.at<cv::Vec3b>(j, i + 1)[0] = color;
                    image.at<cv::Vec3b>(j, i + 1)[1] = color;
                    image.at<cv::Vec3b>(j, i + 1)[2] = color;
                }

                if (j < (int) image.rows - 1) {
                    image.at<cv::Vec3b>(j + 1, i)[0] = color;
                    image.at<cv::Vec3b>(j + 1, i)[1] = color;
                    image.at<cv::Vec3b>(j + 1, i)[2] = color;
                }

                if (j < (int) image.rows - 1 && i < (int) image.cols - 1) {
                    image.at<cv::Vec3b>(j + 1, i + 1)[0] = color;
                    image.at<cv::Vec3b>(j + 1, i + 1)[1] = color;
                    image.at<cv::Vec3b>(j + 1, i + 1)[2] = color;
                }
            }
        }

        imshow("下雪圖", image);

        waitKey(0);
    }

    void showImg_1() {
        double alpha = 0, beta, input;

        Mat img1;
        Mat img2;
        Mat merge;

        cout << "請輸入0~1的數值";
        cin >> input;

        if (alpha >= 0 && alpha <= 1) {
            alpha = input;
        }

        img1 = imread("B:\\bg.jpg");
        img2 = imread("B:\\b.jpg");

        if (!img1.data || !img2.data) {
            cout << "讀不到檔案唷" << endl;
            return;
        }

        namedWindow("合成", CV_LOAD_IMAGE_COLOR);

        beta = (1.0 - alpha);
        addWeighted(img1, alpha, img2, beta, 0.0, merge);

        imshow("合成", merge);

        waitKey(0);
        return;

    }

    void showImg_2() { //Logo
        Mat image = imread("B:\\血小板2.jpg", CV_LOAD_IMAGE_COLOR);
        Mat heart = imread("B:\\heart.png", CV_LOAD_IMAGE_COLOR);

        Mat mergeImg = image, reHeart;

        resize(heart, reHeart, Size(80, 80));

        Mat withHeart;

        withHeart = image(Rect(260, 600, 80, 80));

        addWeighted(withHeart, 0.2, reHeart, 1.0, 0.0, withHeart);

        namedWindow("withHeart");
        imshow("withHeart", image);

        waitKey();

    }

    void showImg_3() { //線性改變亮度

        double alpha = 1;
        int beta = 50;

        Mat image = imread("B:\\血小板.jpg", CV_LOAD_IMAGE_COLOR);

        Mat new_image = Mat::zeros(image.size(), image.type());

        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                for (int c = 0; c < 3; c++) {

                    new_image.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(alpha * (image.at<Vec3b>(i, j)[c]) + beta);
                }
            }
        }

        namedWindow("亮度", 1);

        imshow("亮度", new_image);

        waitKey();
    }

    void showImg_4() {
        Mat atom_image = Mat::zeros(400, 400, CV_8UC3);

//        MyEllipse(atom_image, 0);
//        MyEllipse(atom_image, 45);
//        MyEllipse(atom_image, -45);

        Circle(atom_image);

//        for (int i = 0; i < 36; ++i) {
//            Ellipse(atom_image, 0 + i * 10);
//        }

        namedWindow("test");
        imshow("test", atom_image);

        waitKey(0);

    }

    void showImg_5() {
        Mat image = imread("B:\\血小板.jpg", 1);

        Mat gray_image;

        cvtColor(image, gray_image, CV_BGR2GRAY);

        // 儲存轉換後的圖檔
        imwrite("B:\\灰階血小板.jpg", gray_image);

        // 顯示圖檔視窗大小的控制
        namedWindow("灰階", CV_WINDOW_AUTOSIZE);

        // 顯示灰階圖檔
        imshow("灰階", gray_image);

        waitKey(0);
    }

private:
    void Ellipse(Mat img, double theta) {
        int thickness = 0;

        int col = img.cols;
        int row = img.rows;

        Point p = Point(col / 2, row / 2);
        Size s = Size(img.cols / 4, img.rows / 8);
        Scalar color = Scalar(255, 255, 255);

        ellipse(img, p, s, theta, 0, 360, color, thickness);
    }

    void Circle(Mat img) {
        int thickness = 1;
        int lineType = 1;

        Point p = Point(img.cols / 2, img.rows / 2);

        Size s = Size(img.cols / 2, img.rows / 2);

        Scalar color = Scalar(255, 255, 255);

        ellipse(img, p, s, 0, 0, 360, color, thickness);
    }
};

#endif //OPENCV_SHOWIMAGE_H
