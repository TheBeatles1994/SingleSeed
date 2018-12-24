#ifndef _ALGO_H_
#define _ALGO_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

extern string _myfilename;

/* 种子处理类 */
class SeedProcess
{
public:
    void getSrcMat(string str);
    void getBinary();           //图像二值化。输入原图，输出二值图
    void seedDivision();        //根据面积和凸点个数，区分单个种子与粘连种子
    void getPartSrcMat();       //输出轮廓在原图中对应的矩形区域图
    void getOutline(Scalar color);          //输出标记好轮廓边缘后的图
    Mat getPartBinaryMat(vector<Point> contour);    //输入轮廓，输出该轮廓在二值图中对应的矩形区域图
    int judgeHowMany(Mat& partBinaryMat);           //通过凸点个数来判别种子，输入二值图，输出因种子粘连而产生的凸缺陷个数
private:
    vector<vector<Point>> singleContours;   //单个种子的边界
    Mat srcMat; //原图
    Mat binaryMat;  //二值图
};
/* 测试函数 */
void testSeed(const string str);
/* int转string */
void int2str(const int &int_temp,string &string_temp);
#endif
