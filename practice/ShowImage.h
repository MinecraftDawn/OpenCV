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
        char ig[] = "B:\\��p�O.jpg";

        // ���J����
        Mat image = imread(ig, CV_LOAD_IMAGE_COLOR);

        // �ˬdŪ�ɬO�_���\
        if (!image.data) {
            cout << "�L�k�}�ҩΧ䤣�����" << std::endl;
            return;
        }

        // �إ���ܹ��ɵ���
        namedWindow("Display window", CV_WINDOW_NORMAL);

        // CV_WINDOW_FREERATIO �P CV_WINDOW_KEEPRATIO
        // CV_GUI_NORMAL �P CV_GUI_EXPANDED

        // �b��������ܹ���
        imshow("Display window", image);

        // �������ݫ���
        waitKey(0);
    }

    void showImg2() {
        char ig[] = "B:\\��p�O.jpg";

        // ���J����
        Mat image = imread(ig, CV_LOAD_IMAGE_COLOR);

        imshow("2", image);
        waitKey(0);
    }

    void showImg3() { //�Ƕ�
        char ig[] = "B:\\��p�O.jpg";

        Mat testColor = imread(ig, CV_LOAD_IMAGE_GRAYSCALE);

        namedWindow("Display window", CV_WINDOW_NORMAL);

        imshow("Display window", testColor);

        imwrite("B:\\��p�O��.jpg", testColor);

        waitKey(0);
    }

    void showImg4() {//�����վ�
        char ig[] = "B:\\��p�O.jpg";

        Mat file = imread(ig, CV_LOAD_IMAGE_GRAYSCALE);

        namedWindow("window1", CV_WINDOW_NORMAL);
        namedWindow("window2", CV_WINDOW_AUTOSIZE);

        imshow("window1", file);
        imshow("window2", file);

        resizeWindow("window1", file.cols / 2, file.rows / 2);

        moveWindow("window1", -500, -500);

        waitKey(0);
    }

    void showImg5() { //�ۭq�Ƕ�
        char ig[] = "B:\\��p�O.jpg";

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
        char ig[] = "B:\\��p�O.jpg";

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

    void showImg7() {
        char ig[] = "B:\\��p�O.jpg";

        Mat file = imread(ig, CV_LOAD_IMAGE_COLOR);

        Mat splitC[3];

        split(file, splitC);

        imshow("B", splitC[0]);
        imshow("G", splitC[1]);
        imshow("R", splitC[2]);

        splitC[0] = Mat::zeros(splitC[0].size(), CV_8UC1);
//        splitC[1] = Mat::zeros(splitC[1].size(),CV_8UC1);
        splitC[2] = Mat::zeros(splitC[2].size(), CV_8UC1);

        Mat output;

        merge(splitC, 3, output);

        imshow("Merged", output);

        waitKey(0);
    }

    void showImg8() {
        char ig[] = "B:\\��p�O.jpg";

        Mat file = imread(ig, CV_LOAD_IMAGE_COLOR);

        Mat originalFloat;

        file.convertTo(originalFloat, CV_32FC1, 1.0 / 255.0);

        Mat dftOriginal;

        Mat originalComplex[2] = {originalFloat,Mat::zeros(originalComplex->size(),CV_32F)};

        Mat dftReady;

        merge(originalComplex,2,dftReady);

        dft(dftReady,dftOriginal,DFT_COMPLEX_OUTPUT);

        Mat splitArray[2] = {Mat::zeros(dftOriginal.size(),CV_32F),Mat::zeros(dftOriginal.size(),CV_32F)};

        split(dftOriginal,splitArray);

        Mat dftMagnitude;

        magnitude(splitArray[0],splitArray[1],dftMagnitude);

        dftMagnitude += Scalar::all(1);

        log(dftMagnitude,dftMagnitude);

        normalize(dftMagnitude,dftMagnitude,0,1,CV_MINMAX);

        imshow("DFT",dftMagnitude);

        waitKey(0);



    }

};


#endif //OPENCV_SHOWIMAGE_H
