
#include <iostream>

#include "practice/ShowImage.h"

using namespace std;

int main(void) {
    ShowImage *s = new ShowImage();
    s->showImg_13();
    return 0;
}

//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <math.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
///// �ŧi�����ܼ�
//
//const char* window_name = "Pyramids Demo";
//
//int main(void)
//{
//    Mat src, dst, tmp;
//    /// �{���ϥλ���
//    printf("\n �Ϲ��Y��ܽd\n ");
//    printf("------------------ \n");
//    printf(" * [u] -> ��j \n");
//    printf(" * [d] -> �Y�p \n");
//    printf(" * [ESC] -> �����{�� \n \n");
//
//    src = imread("B:\\��p�O.jpg");
//    if (!src.data)
//    {
//        printf(" No data! -- Exiting the program \n");
//        return -1;
//    }
//
//    // �N�r�J�ϩ�J�n�B�z���ܼƤ�
//    tmp = src;
//    dst = tmp;
//
//    /// �إߵ���
//    namedWindow(window_name, WINDOW_AUTOSIZE);
//    imshow(window_name, dst);
//
//    /// Loop
//    for (;;)
//    {
//        int c;
//        c = waitKey(10);
//
//        if ((char)c == 27)
//            break;
//
//        if ((char)c == 'u')
//        {
//            pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
//            printf("** ��j: ��j�⭿\n");
//        } else if ((char)c == 'd') {
//            pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
//            printf("** �Y�p: �Y�p�@�b\n");
//        }
//
//        imshow(window_name, dst);
//
//        // �N���G���n�B�z����
//        tmp = dst;
//    }
//
//    return 0;
//}
