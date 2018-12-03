#ifndef _ALGO_H_
#define _ALGO_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

/* 主要的种子处理类 */
class SeedProcess
{
public:
    void getSrcMat(string str);
    void getBinary();              //图像二值化。输入原图，输出二值图
    void areaDivision();      //根据面积和凸点个数，区分单个种子与粘连种子
    void getPartSrcMat();
private:
    Mat getPartBinaryMat(vector<Point> contour);
    int judgeHowMany(Mat& partBinaryMat);

    vector<vector<Point>> singleContours;   //单个种子的边界
    Mat srcMat; //原图
    Mat binaryMat;  //二值图
};

void testSeed(const string str);
void int2str(const int &int_temp,string &string_temp);
#endif
