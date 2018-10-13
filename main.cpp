
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
///// 宣告全域變數
//
//const char* window_name = "Pyramids Demo";
//
//int main(void)
//{
//    Mat src, dst, tmp;
//    /// 程式使用說明
//    printf("\n 圖像縮放示範\n ");
//    printf("------------------ \n");
//    printf(" * [u] -> 放大 \n");
//    printf(" * [d] -> 縮小 \n");
//    printf(" * [ESC] -> 結束程式 \n \n");
//
//    src = imread("B:\\血小板.jpg");
//    if (!src.data)
//    {
//        printf(" No data! -- Exiting the program \n");
//        return -1;
//    }
//
//    // 將毒入圖放入要處理的變數中
//    tmp = src;
//    dst = tmp;
//
//    /// 建立視窗
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
//            printf("** 放大: 放大兩倍\n");
//        } else if ((char)c == 'd') {
//            pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
//            printf("** 縮小: 縮小一半\n");
//        }
//
//        imshow(window_name, dst);
//
//        // 將結果當成要處理的圖
//        tmp = dst;
//    }
//
//    return 0;
//}
