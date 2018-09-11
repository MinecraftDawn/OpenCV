#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(void)
{

    char ig[] = "B:\\��p�O.jpg";

    Mat image;

    // ���J����
    image = imread(ig, CV_LOAD_IMAGE_COLOR);

    // �ˬdŪ�ɬO�_���\
    if (!image.data)
    {
        cout << "�L�k�}�ҩΧ䤣�����" << std::endl;
        return -1;
    }

    // �إ���ܹ��ɵ���
    namedWindow("Display window", CV_WINDOW_NORMAL);

    // CV_WINDOW_FREERATIO �P CV_WINDOW_KEEPRATIO
    // CV_GUI_NORMAL �P CV_GUI_EXPANDED

    // �b��������ܹ���
    imshow("Display window", image);

    // �������ݫ���
    waitKey(0);


    return 0;
}