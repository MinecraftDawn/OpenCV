//
// Created by Eric on 2018/9/12.
//

#ifndef OPENCV_SHOWIMAGE_H
#define OPENCV_SHOWIMAGE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv\cv.h>
#include <iostream>
#include <stdint.h>
#include <random>
#include <ctime>
#include <vector>

using namespace cv;
using namespace std;

struct userdata {
    Mat src;
    int trackbarValue;
    int trackbarType;

};

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
                file.at<Vec3b>(i, j)[2] = file.at<Vec3b>(i, j)[2] * 0;//R
            }
        }

        imshow("Display", file);
        imshow("D2", file2);
        waitKey(0);

    }

    void showImg7() { //merge
        char ig[] = "B:\\��p�O.jpg";

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
        char ig[] = "B:\\��p�O.jpg";

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
        char img[] = "B:\\��p�O.jpg";

        Mat image;

        // ���J����
        image = imread(img, CV_LOAD_IMAGE_COLOR);

        // �ˬdŪ�ɬO�_���\
        if (!image.data) {
            cout << "�L�k�}�ҩΧ䤣�����" << std::endl;
            return;
        }

        // �إ���ܹ��ɵ���
        namedWindow("���", CV_WINDOW_NORMAL);
        namedWindow("�U����", CV_WINDOW_NORMAL);

        imshow("���", image);

        // ���I��
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

        imshow("�U����", image);

        waitKey(0);
    }

    void showImg_1() {
        double alpha = 0, beta, input;

        Mat img1;
        Mat img2;
        Mat merge;

        cout << "�п�J0~1���ƭ�";
        cin >> input;

        if (alpha >= 0 && alpha <= 1) {
            alpha = input;
        }

        img1 = imread("B:\\bg.jpg");
        img2 = imread("B:\\b.jpg");

        if (!img1.data || !img2.data) {
            cout << "Ū�����ɮ׭�" << endl;
            return;
        }

        namedWindow("�X��", CV_LOAD_IMAGE_COLOR);

        beta = (1.0 - alpha);
        addWeighted(img1, alpha, img2, beta, 0.0, merge);

        imshow("�X��", merge);

        waitKey(0);
        return;

    }

    void showImg_2() { //Logo
        Mat image = imread("B:\\��p�O2.jpg", CV_LOAD_IMAGE_COLOR);
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

    void showImg_3() { //�u�ʧ��ܫG��

        double alpha = 1;
        int beta = 50;

        Mat image = imread("B:\\��p�O.jpg", CV_LOAD_IMAGE_COLOR);

        Mat new_image = Mat::zeros(image.size(), image.type());

        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                for (int c = 0; c < 3; c++) {

                    new_image.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(alpha * (image.at<Vec3b>(i, j)[c]) + beta);
                }
            }
        }

        namedWindow("�G��", 1);

        imshow("�G��", new_image);

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
        Mat image = imread("B:\\��p�O.jpg", 1);

        Mat gray_image;

        cvtColor(image, gray_image, CV_BGR2GRAY);

        // �x�s�ഫ�᪺����
        imwrite("B:\\�Ƕ���p�O.jpg", gray_image);

        // ��ܹ��ɵ����j�p������
        namedWindow("�Ƕ�", CV_WINDOW_AUTOSIZE);

        // ��ܦǶ�����
        imshow("�Ƕ�", gray_image);

        waitKey(0);
    }

    void showImg_6() {
        Mat atom_image = Mat::zeros(400, 400, CV_8UC3);

        for (int i = 0; i < 36; ++i) {
            Ellipse(atom_image, 0 + i * 10);
        }

        namedWindow("test");
        imshow("test", atom_image);

        waitKey(0);

    }

    void showImg_7() {
        Mat atom_image = Mat::zeros(400, 400, CV_8UC3);
        Scalar color = Scalar(255, 255, 255);

        Point p[2][2];
        p[0][0] = Point(100, 200);
        p[0][1] = Point(300, 200);
        p[1][0] = Point(200, 100);
        p[1][1] = Point(200, 300);

        for (int i = 0; i < 2; ++i) {
            line(atom_image, p[i][0], p[i][1], color, 1);
        }

        namedWindow("test");
        imshow("test", atom_image);

        waitKey(0);

    }

    void showImg_8() {
        Mat atom_image = Mat::zeros(800, 800, CV_8UC3);
        Scalar color = Scalar(255, 255, 255);

        Point center = Point(atom_image.cols / 2, atom_image.rows / 2);
        Scalar white = Scalar(255, 255, 255);

        double inRadius;
        double outRadius;

        /*
         * �~��
         */
        int circleNum = 3;
        for (int i = 0; i < circleNum; ++i) {
            double size = 2 + 0.15 * i;
            Size cirSize = Size(atom_image.cols / size, atom_image.rows / size);

            ellipse(atom_image, center, cirSize, 0, 0, 360, white, 1);
            inRadius = atom_image.cols / size;

            /*
             * �ꤺ�e�u
             */
            if (i == circleNum - 1) {
                double preCirRadius = atom_image.cols / (2 + 0.15 * (circleNum - 2));
                double curCirRadius = atom_image.cols / (2 + 0.15 * (circleNum - 1));

                Point2d prePoint, curPoint;

                int cirLineNum = 72;
                for (int j = 0; j < cirLineNum; ++j) {
                    prePoint = Point2d(center.x + cos(2 * CV_PI / cirLineNum * j) * preCirRadius,
                                       center.y + sin(2 * CV_PI / cirLineNum * j) * preCirRadius);
                    curPoint = Point2d(center.x + cos(2 * CV_PI / cirLineNum * j) * curCirRadius,
                                       center.y + sin(2 * CV_PI / cirLineNum * j) * curCirRadius);

                    line(atom_image, prePoint, curPoint, white);
                }
            }

            if (i == 0) {
                outRadius = atom_image.cols / size;
            }
        }

        /*
         * ���������
         */
        Point squarePoint;
        Point2d outSquarePoint;
        double InnerRadius;
        for (int i = 0; i < 3; ++i) {
            Point square1[4];
            double Degress = 45 + 30 * i;

            for (int j = 0; j < 2; ++j) {
                InnerRadius = inRadius * (1 - 0.02 * j);

                square1[0] = Point(center.x + cos(Degress / 180 * CV_PI) * InnerRadius,
                                   center.y + sin(Degress / 180 * CV_PI) * InnerRadius);
                square1[1] = Point(center.x + cos((Degress + 90) / 180 * CV_PI) * InnerRadius,
                                   center.y + sin((Degress + 90) / 180 * CV_PI) * InnerRadius);
                square1[2] = Point(center.x + cos((Degress + 180) / 180 * CV_PI) * InnerRadius,
                                   center.y + sin((Degress + 180) / 180 * CV_PI) * InnerRadius);
                square1[3] = Point(center.x + cos((Degress + 270) / 180 * CV_PI) * InnerRadius,
                                   center.y + sin((Degress + 270) / 180 * CV_PI) * InnerRadius);

                for (int k = 0; k < 3; ++k) {
                    line(atom_image, square1[k], square1[k + 1], white, 1);
                }
                line(atom_image, square1[3], square1[0], white, 1);

                if (j == 0) {
                    squarePoint = square1[1];
                    outSquarePoint = Point(center.x + cos((Degress + 90) / 180 * CV_PI) * outRadius,
                                           center.y + sin((Degress + 90) / 180 * CV_PI) * outRadius);
                }
            }
        }

        /*
         * ����+�e�u
         */
        for (int i = 0; i < 2; ++i) {
            double squInnerCirRadius = InnerRadius / (1 + 0.05 * i) / sqrt(2);
            Size innerCirSize = Size(squInnerCirRadius, squInnerCirRadius);

            ellipse(atom_image, center, innerCirSize, 0, 0, 360, white);

            if (i == 0) {
                double preCirRadius = InnerRadius / (1 + 0.05 * i) / sqrt(2);
                double curCirRadius = InnerRadius / (1 + 0.05 * (i + 1)) / sqrt(2);
                Point2d prePoint, curPoint;

                int cirLineNum = 72;
                for (int j = 0; j < cirLineNum; ++j) {
                    prePoint = Point2d(center.x + cos(2 * CV_PI / cirLineNum * j) * preCirRadius,
                                       center.y + sin(2 * CV_PI / cirLineNum * j) * preCirRadius);
                    curPoint = Point2d(center.x + cos(2 * CV_PI / cirLineNum * j) * curCirRadius,
                                       center.y + sin(2 * CV_PI / cirLineNum * j) * curCirRadius);

                    line(atom_image, prePoint, curPoint, white);
                }
            }
        }

        /*
         * ���~�P(�e)
         */
        double startDistance = InnerRadius / sqrt(2);
        Point2d startOutAngle[5];
        Point2d startInAngle[5];

        int angleNum = 5;
        for (int i = 0; i < angleNum; ++i) {
            startOutAngle[i] = Point2d(
                    center.x + cos(2 * CV_PI / angleNum * i) * startDistance,
                    center.y + sin(2 * CV_PI / angleNum * i) * startDistance);

            startInAngle[i] = Point2d(
                    center.x + cos(2 * CV_PI / angleNum * i +
                                   (double(1) / 5) * CV_PI) * startDistance / 5 * 2,

                    center.y + sin(2 * CV_PI / angleNum * i +
                                   (double(1) / 5) * CV_PI) * startDistance / 5 * 2);
        }
        for (int i = 0; i < angleNum; ++i) {
            line(atom_image, startOutAngle[i], startInAngle[i], white);

            line(atom_image, startOutAngle[(i + 1) % 5], startInAngle[i % 5], white);
        }


        /*
        * ���~�P(��)
        */
        double starRotate = double(2) / 10;
        Point2d startOutAngle2[5];
        Point2d startInAngle2[5];

        for (int i = 0; i < angleNum; ++i) {
            startOutAngle2[i] = Point2d(
                    center.x + cos(2 * CV_PI / angleNum * i + starRotate * CV_PI) * startDistance,
                    center.y + sin(2 * CV_PI / angleNum * i + starRotate * CV_PI) * startDistance);

            startInAngle2[i] = Point2d(
                    center.x + cos(2 * CV_PI / angleNum * i +
                                   (starRotate + (double(1) / 5)) * CV_PI) * startDistance / 5 * 2,

                    center.y + sin(2 * CV_PI / angleNum * i +
                                   (starRotate + (double(1) / 5)) * CV_PI) * startDistance / 5 * 2);
        }

        //���o�J�I�A�åB�e�u(�P�P��t*2&�����u)
        for (int i = 0; i < angleNum; ++i) {
            Point2d intersection1 = getIntersection(startOutAngle[i],
                                                    startInAngle[i],
                                                    startOutAngle2[i],
                                                    startInAngle2[(i + 4) % 5]);

            Point2d intersection2 = getIntersection(startOutAngle[(i + 1) % 5],
                                                    startInAngle[i],
                                                    startOutAngle2[i],
                                                    startInAngle2[i]);

            line(atom_image, intersection1, startOutAngle2[i], white);
            line(atom_image, intersection2, startOutAngle2[i], white);
            line(atom_image, startInAngle[i], startOutAngle2[i], white);
        }




        /*
         * ��G(��)
         */
        Point moonInnerCenter = center + (squarePoint - center) / 4 * 3;
        double moonInnerRadius = getPointDistance(squarePoint, moonInnerCenter);

        Size moonInnerSize = Size(moonInnerRadius, moonInnerRadius);
        ellipse(atom_image, moonInnerCenter, moonInnerSize, 0, 0, 360, white);

        /*
         * ��G(�~)
         */
        Point2d unitVec = getUnitVec(moonInnerCenter, center);
        Point2d moonInnerEdge = (Point2d) moonInnerCenter + unitVec * moonInnerRadius;
        Point2d moonOutCenter = Point2d(moonInnerEdge + outSquarePoint) / 2;

        double moonOutRadius = getPointDistance(moonInnerEdge, moonOutCenter);
        Size moonOutSize = Size(moonOutRadius, moonOutRadius);

        ellipse(atom_image, moonOutCenter, moonOutSize, 0, 0, 360, white);

        namedWindow("�]�k�}");
        imshow("�]�k�}", atom_image);

        imwrite("�]�k�}.png", atom_image);

        waitKey(0);
    }

    void showImg_9() {
        Mat image = imread("B:\\��p�O.jpg", 1);

        Mat hsv;

        cvtColor(image, hsv, CV_BGR2HSV);

        namedWindow("www", CV_WINDOW_AUTOSIZE);

        imshow("www", hsv);

        waitKey(0);

    }

    void showImg_10() {
        Mat image = imread("B:\\lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);

        srand(time(NULL));
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                unsigned int color = image.at<uint8_t>(i, j);

                image.at<uint8_t>(i, j) = (color - 30) * 1.3;
            }
        }

        Mat ch;
        int hs = 256;
        float range[] = {0, 255};
        const float *hisRange = {range};
        calcHist(&image, 1, 0, Mat(), ch, 1, &hs, &hisRange);
        Mat showHistImg(256, 256, CV_8UC1, Scalar(255));

        drawHistImg(ch, showHistImg);
        imshow("�����", showHistImg);
        imshow("Display", image);
        waitKey(0);

    }

    void showImg_11() {//�ҽk����
        Mat dst;
        int KERNEL_LENGTH = 10;
        if (KERNEL_LENGTH % 2 == 0) KERNEL_LENGTH++;
        Mat img = imread("B:\\lena.jpg", CV_LOAD_IMAGE_COLOR);

        // �ˬdŪ�ɬO�_���\
        if (!img.data) {
            return;
        }

        int per = 10;
        int i = img.cols * img.rows;

        for (int k = 0; k < i * per / 100; k++) {

            int i = rand() % img.cols;
            int j = rand() % img.rows;
            if (img.channels() == 3) {
                img.at<Vec3b>(j, i)[0] = 255;
                img.at<Vec3b>(j, i)[1] = 255;
                img.at<Vec3b>(j, i)[2] = 255;
            }
        }

        dst = img.clone();

        ///�ϥ� Median �ҽk�k
        medianBlur(img, dst, KERNEL_LENGTH);
        imshow("Median���Ƽҽk�k", dst);

        imshow("10%", img);
        waitKey(0);
    }

    void showImg_12() {//����
        Mat img = imread("B:\\opencv-logo.png");
        Mat dst;

        int dilation_size = 20;
        Mat element = getStructuringElement(MORPH_CROSS,
                                            Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                            Point(dilation_size, dilation_size));
        erode(img, dst, element);

        Point textOrg(10, 25);
        putText(dst, "", textOrg, FONT_HERSHEY_SIMPLEX, 1, 3);

        imshow("����", dst);

        waitKey(0);
    }

    void showImg_13() {
        Mat src, dst, tmp;

        src = imread("B:\\��p�O.jpg");

        if (!src.data) {
            cout << "�L�kŪ�����ɡI" << endl;
            return;
        }

        cout << "��[u] ��j" << endl;
        cout << "��[d] �Y�p" << endl;
        cout << "��[ESC] �����{��" << endl;

        tmp = dst = src;

        namedWindow("�Y��", WINDOW_AUTOSIZE);
        imshow("�Y��", dst);

        char c = 0;
        while (true) {
            c = waitKey(0);

            if (c == 27) {
                break;
            } else if (c == 'u') {
                if (tmp.cols < src.cols) {
                    int multiple = (src.cols / tmp.cols);
                    tmp = src;

                    if (multiple == 2) {
                        dst = src;
                    }

                    while (multiple / 2 != 1) {
                        multiple /= 2;
                        pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
                        tmp = dst;
                    }


                } else {
                    pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
                }
                cout << "��j" << endl;
            } else if (c == 'd') {
                pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
                cout << "�Y�p" << endl;
            }

            imshow("�Y��", dst);
            tmp = dst;
        }
    }

    void showImg_14() {
        Mat src;
        src = imread("B:\\chicky_512.png", 1);
        int trackbarValue = 0;
        int trackbarType = 3;
        int maxType = 4;

        namedWindow("�{�ɭ�", WINDOW_AUTOSIZE);

        imshow("�{�ɭ�", src);

        Mat dst;
        threshold(src, dst, 100,
                  255, trackbarType);

        imshow("�{�ɭ�", dst);

        waitKey(0);
    }

    void showImg_15() {
        Mat src, dst;
        int broad;
        int borderType;

        RNG rng(12345);

        src = imread("B:\\lena.jpg");

        if (!src.data)
            return;

        cout << "�� 'c' �H���]�w���" << endl;
        cout << "�� 'r' �������" << endl;
        cout << "�� 'ESC' �����{��" << endl;

        namedWindow("�ƻs���", WINDOW_AUTOSIZE);


        broad = 0.1 * src.rows;

        dst = src;

        imshow("�ƻs���", dst);

        while (true) {
            int c = waitKey(0);

            if ((char) c == 27) {
                break;
            } else if ((char) c == 'c') {
                // �ɩT�w��
                borderType = BORDER_CONSTANT;
            } else if ((char) c == 'r') {
                // ��ϭȥ[��
                borderType = BORDER_REPLICATE;
            }

            Scalar value(rng.uniform(0, 255), rng.uniform(0, 255),
                         rng.uniform(0, 255));

            copyMakeBorder(src, dst, broad, broad, broad, broad,
                           borderType, value);

            imshow("�ƻs���", dst);
        }
    }

    void showImg_16() {
        Mat src, gray;
        Mat sobelX, sobelY, absX1, absY1, sobelFinal;
        Mat cannyX, cannyY, absX2, absY2, cannyFinal;

        int ksize = 3, dx = 1, dy = 1;
        double scale = 5;

        src = imread("B:\\��p�O.jpg", CV_LOAD_IMAGE_COLOR);

        if (!src.data) return;

        GaussianBlur(src, src, Size(3, 3), 0, 0);

        cvtColor(src, gray, COLOR_RGB2GRAY);

        //��X�L������t
        Sobel(gray, sobelX, CV_16U, dx, 0, ksize, scale);
        convertScaleAbs(sobelX, absX1);

        Sobel(gray, sobelY, CV_16U, 0, dy, ksize, scale);
        convertScaleAbs(sobelY, absY1);

        addWeighted(absX1, 0.5, absY1, 0.5, 0, sobelFinal);

        imshow("sobel", sobelFinal);


        Canny(gray, cannyX, 50, 200, 3);
        convertScaleAbs(cannyX, absX2);

        Canny(gray, cannyY, 50, 200, 3);
        convertScaleAbs(cannyY, absY2);

        addWeighted(absX2, 0.5, absY2, 0.5, 0, cannyFinal);

        imshow("canny", cannyFinal);

        waitKey(0);
    }

    void showImg_17() { //Close
        int size = 1;
        Mat dst;
        Mat src = imread("B:\\���.png", CV_LOAD_IMAGE_COLOR);

        Mat element = getStructuringElement(1, Size(2 * size + 1, 2 * size + 1),
                                            Point(size, size));

        dilate(src, dst, element,Point(1,1),2);
        erode(dst, dst, element,Point(1,1),1);

        dilate(dst, dst, element,Point(1,1),1);
        erode(dst, dst, element,Point(1,1),2);


        imshow("�٨SClose",src);
        imshow("Close",dst);
        waitKey(0);
    }

private:

    void Ellipse(Mat img, double theta) {//���
        int thickness = 0;

        int col = img.cols;
        int row = img.rows;

        Point p = Point(col / 2, row / 2);
        Size s = Size(img.cols / 4, img.rows / 8);
        Scalar color = Scalar(255, 255, 255);

        ellipse(img, p, s, theta, 0, 360, color, thickness);
    }

    void Circle(Mat img) {//�ꫬ
        int thickness = 1;
        int lineType = 1;

        Point p = Point(img.cols / 2, img.rows / 2);

        Size s = Size(img.cols / 2, img.rows / 2);

        Scalar color = Scalar(255, 255, 255);

        ellipse(img, p, s, 0, 0, 360, color, thickness);
    }

    double getPointDistance(Point p1, Point p2) {//���o�I�������Z��
        Point subPoint = p1 - p2;
        return sqrt(pow(subPoint.x, 2) + pow(subPoint.y, 2));
    }

    double getPointDistance(Point2d p1, Point2d p2) {//���o�I�������Z��
        Point2d subPoint = p1 - p2;
        return sqrt(pow(subPoint.x, 2) + pow(subPoint.y, 2));
    }

    Point2d getUnitVec(Point p1, Point p2) {//���o���V�q
        Point2d unit;

        unit = p2 - p1;
        double distance = getPointDistance(p1, p2);

        unit.x = unit.x / distance;
        unit.y = unit.y / distance;

        return unit;
    }

    Point2d getIntersection(Point2d p1, Point2d p2, Point2d p3, Point2d p4) {

        double m1, m2, b1, b2;
        m1 = (p1.x - p2.x == 0) ? 0 : (p1.y - p2.y) / (p1.x - p2.x);
        b1 = p1.y - p1.x * m1;

        m2 = (p3.x - p4.x == 0) ? 0 : (p3.y - p4.y) / (p3.x - p4.x);
        b2 = p3.y - p3.x * m2;

        //m1x-y=-b1
        //m2x-y=-b2
        double delta, deltaX, deltaY;
        delta = m1 * (-1) - (-1) * m2;
        deltaX = -b1 * (-1) - (-1) * -b2;
        deltaY = m1 * (-b2) - (-b1) * m2;

        if (delta == 0) {
            return Point2d(0, 0);
        } else {
            return Point2d(deltaX / delta, deltaY / delta);
        }

    }

    void drawHistImg(const Mat &src, Mat &dst) { //�e�����
        int histSize = 256;
        float histMaxValue = 0;
        for (int i = 0; i < histSize; i++) {
            float tempValue = src.at<float>(i);
            if (histMaxValue < tempValue) {
                histMaxValue = tempValue;
            }
        }

        float scale = (0.9 * 256) / histMaxValue;
        for (int i = 0; i < histSize; i++) {
            int intensity = static_cast<int>(src.at<float>(i) * scale);
            line(dst, Point(i, 255), Point(i, 255 - intensity), Scalar(0));
        }
    }
};

#endif //OPENCV_SHOWIMAGE_H
