#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(void)
{

    char ig[] = "B:\\血小板.jpg";

    Mat image;

    // 載入圖檔
    image = imread(ig, CV_LOAD_IMAGE_COLOR);

    // 檢查讀檔是否成功
    if (!image.data)
    {
        cout << "無法開啟或找不到圖檔" << std::endl;
        return -1;
    }

    // 建立顯示圖檔視窗
    namedWindow("Display window", CV_WINDOW_NORMAL);

    // CV_WINDOW_FREERATIO 與 CV_WINDOW_KEEPRATIO
    // CV_GUI_NORMAL 與 CV_GUI_EXPANDED

    // 在視窗內顯示圖檔
    imshow("Display window", image);

    // 視窗等待按鍵
    waitKey(0);


    return 0;
}