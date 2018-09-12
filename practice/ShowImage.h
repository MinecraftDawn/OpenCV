//
// Created by Eric on 2018/9/12.
//

#ifndef OPENCV_SHOWIMAGE_H
#define OPENCV_SHOWIMAGE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;


class ShowImage {
public:

    void showImg(){
        char ig[] = "B:\\��p�O.jpg";

        // ���J����
        Mat image = imread(ig, CV_LOAD_IMAGE_COLOR);

        // �ˬdŪ�ɬO�_���\
        if (!image.data)
        {
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

    void showImg2(){
        char ig[] = "B:\\��p�O.jpg";

        // ���J����
        Mat image = imread(ig, CV_LOAD_IMAGE_COLOR);

        imshow("2",image);
        waitKey(0);
    }

    void showImg3(){
        char ig[] = "B:\\��p�O.jpg";

        Mat testColor = imread(ig,CV_LOAD_IMAGE_GRAYSCALE);

        namedWindow("Display window", CV_WINDOW_NORMAL);

        imshow("Display window",testColor);

        imwrite("B:\\��p�O��.jpg",testColor);

        waitKey(0);
    }

};


#endif //OPENCV_SHOWIMAGE_H
