#include "algo.h"
#include <sstream>

string _myfilename = "FB057_2.jpg";
/*
 * @函数功能:测试程序
 * @输入参数:图片地址字符串
 * @输出参数:无
 * @注意事项:
 *      无
 */
void testSeed(const string str)
{
    /* 读取原图片 */
    SeedProcess seed;
    /* 读取原图 */
    seed.getSrcMat(str);
    /* 二值化 */
    seed.getBinary();
    /* 提取单个种子边界点 */
    seed.seedDivision();
    //seed.getPartSrcMat();
    seed.getOutline(Scalar(255,255,255));

}

void SeedProcess::getSrcMat(string str)
{
    srcMat = imread(str);
}

/*
* 函数功能：
* 图像二值化。输入原图，输出二值图
*/
#define SEEDGRAYVALUE 25               //定义灰度值阈值
void SeedProcess::getBinary()
{
    Mat tempGrayMat = imread("srcimage/FB057_2_binary.jpg");
    //cv::cvtColor(srcMat, tempGrayMat, CV_BGR2GRAY);
    cv::cvtColor(tempGrayMat, tempGrayMat, CV_BGR2GRAY);
    threshold(tempGrayMat, binaryMat, SEEDGRAYVALUE, 255, CV_THRESH_BINARY);
    //threshold(tempGrayMat, binaryMat, SEEDGRAYVALUE, 255, CV_THRESH_BINARY_INV);
    Mat diamond = Mat(3, 3, CV_8UC1, cv::Scalar(1));
//    morphologyEx(binaryMat, binaryMat, MORPH_ERODE, diamond);
    morphologyEx(binaryMat, binaryMat, MORPH_OPEN, diamond);
    morphologyEx(binaryMat, binaryMat, MORPH_OPEN, diamond);
//    morphologyEx(binaryMat, binaryMat, MORPH_CLOSE, diamond);
//    morphologyEx(binaryMat, binaryMat, MORPH_CLOSE, diamond);
//    Canny(binaryMat, binaryMat, 50,100);
    //imwrite("srcimage/FB057_2_binarys.jpg", binaryMat);
}

/*
* 函数功能：
* 二值图轮廓提取。输入二值图，输出全部轮廓
*/
vector<vector<Point>> getContours(Mat binaryMat)
{
    vector<vector<Point>> temContours;
    vector<Vec4i> temHeichy;
    cv::findContours(binaryMat, temContours, temHeichy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
    return temContours;
}

/*
* 函数功能：
* 根据面积和凸点个数，初步区分单个种子与粘连种子，单个种子存入signelContours中，粘连种子存入complexContours中
*/
void SeedProcess::seedDivision()
{
    //得到所有边界点
    vector<vector<Point>> contours = getContours(binaryMat);
    //对每一个边界点进行处理
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area> 50 && area < 2500)
        {
            singleContours.push_back(contours[i]);
        }
        else if (area >= 2500)
        {
                Mat partBinaryMat = getPartBinaryMat(Mat(contours[i]));
                if (judgeHowMany(partBinaryMat) == 0)         //因种子粘连而产生的凸缺陷个数为0时
                {
                    singleContours.push_back(contours[i]);         //认为粘连区域是单个种子，并将单个种子轮廓加入signelContours中
                }
        }
    }
#if MYDEBUG
    cout << "剩余个数=" << complexContours.size() << endl;
#endif
}


/*
* 函数功能：
* 输入轮廓，输出该轮廓在二值图中对应的矩形区域图
*/
Mat SeedProcess::getPartBinaryMat(vector<Point> contour)
{
    Rect rect = cv::boundingRect(Mat(contour));
    rect.x -= 5;
    rect.y -= 5;
    rect.width += 10;
    rect.height += 10;
    Mat tempMat = Mat::zeros(rect.size(), CV_8UC1);
    //cout << "tempMat type:" << temMat.type() << endl;
    //cout << "binaryMat type:" << binaryMat.type() << endl;
    for (int row = 0; row < tempMat.rows; row++)       //将粘连区域从二值图中copy下来,复制给图像t
    {
        for (int col = 0; col < tempMat.cols; col++)
        {
            Point pt;
            pt.x = rect.x + col;
            pt.y = rect.y + row;
            if (pointPolygonTest(contour, pt, false) >= 0)     //pointPolygonTest函数，判断点是否在轮廓内，测试点在轮廓内、外、边上时，返回值分别为正、负、0
            {
                tempMat.at<uchar>(row, col) = binaryMat.at<uchar>(pt);        //当外接矩形内的点在粘连区域轮廓内时，复制该点像素值
            }
        }
    }
    return tempMat;
}

/*
* 函数功能：
* 输出轮廓在原图中对应的矩形区域图
*/
void SeedProcess::getPartSrcMat()
{
    int count = 0;
    for(vector<Point> contour:singleContours)
    {
        Rect rect = cv::boundingRect(Mat(contour));           //轮廓的垂直边界最小矩形
        rect.x -= 5;
        rect.y -= 5;
        rect.width += 10;
        rect.height += 10;
        Mat tempMat = Mat(rect.size(), CV_8UC3, Scalar(0, 0, 0));
        for (int row = 0; row < tempMat.rows; row++)       //将粘连区域从二值图中copy下来,复制给图像t
        {
            for (int col = 0; col < tempMat.cols; col++)
            {
                Point pt;
                pt.x = rect.x + col;
                pt.y = rect.y + row;
                if (pointPolygonTest(contour, pt, false) >= 0)     //pointPolygonTest函数，判断点是否在轮廓内，测试点在轮廓内、外、边上时，返回值分别为正、负、0
                {
                    tempMat.at<Vec3b>(row, col)[0] = srcMat.at<Vec3b>(pt.y, pt.x)[0];
                    tempMat.at<Vec3b>(row, col)[1] = srcMat.at<Vec3b>(pt.y, pt.x)[1];       //复制rgb值
                    tempMat.at<Vec3b>(row, col)[2] = srcMat.at<Vec3b>(pt.y, pt.x)[2];
                }
            }
        }
        //imshow("test", tempMat);
        //waitKey();
        string str;
        int2str(count, str);
        imwrite("outimage/" + _myfilename + "_" + str + ".jpg", tempMat);
        count++;
    }
}

void SeedProcess::getOutline(Scalar color)
{
    int count = 0;
    for(vector<Point> contour:singleContours)
    {
        for(Point pt:contour)
        {
            srcMat.at<Vec3b>(pt.y, pt.x)[0] = color[0];
            srcMat.at<Vec3b>(pt.y, pt.x)[1] = color[1];
            srcMat.at<Vec3b>(pt.y, pt.x)[2] = color[2];

            //将其八邻域都变红，增加边缘线宽度
            srcMat.at<Vec3b>(pt.y+1, pt.x-1)[0] = color[0];
            srcMat.at<Vec3b>(pt.y+1, pt.x-1)[1] = color[1];
            srcMat.at<Vec3b>(pt.y+1, pt.x-1)[2] = color[2];
            srcMat.at<Vec3b>(pt.y+1, pt.x)[0] = color[0];
            srcMat.at<Vec3b>(pt.y+1, pt.x)[1] = color[1];
            srcMat.at<Vec3b>(pt.y+1, pt.x)[2] = color[2];
            srcMat.at<Vec3b>(pt.y+1, pt.x+1)[0] = color[0];
            srcMat.at<Vec3b>(pt.y+1, pt.x+1)[1] = color[1];
            srcMat.at<Vec3b>(pt.y+1, pt.x+1)[2] = color[2];
            srcMat.at<Vec3b>(pt.y, pt.x-1)[0] = color[0];
            srcMat.at<Vec3b>(pt.y, pt.x-1)[1] = color[1];
            srcMat.at<Vec3b>(pt.y, pt.x-1)[2] = color[2];
            srcMat.at<Vec3b>(pt.y, pt.x+1)[0] = color[0];
            srcMat.at<Vec3b>(pt.y, pt.x+1)[1] = color[1];
            srcMat.at<Vec3b>(pt.y, pt.x+1)[2] = color[2];
            srcMat.at<Vec3b>(pt.y-1, pt.x-1)[0] = color[0];
            srcMat.at<Vec3b>(pt.y-1, pt.x-1)[1] = color[1];
            srcMat.at<Vec3b>(pt.y-1, pt.x-1)[2] = color[2];
            srcMat.at<Vec3b>(pt.y-1, pt.x)[0] = color[0];
            srcMat.at<Vec3b>(pt.y-1, pt.x)[1] = color[1];
            srcMat.at<Vec3b>(pt.y-1, pt.x)[2] = color[2];
            srcMat.at<Vec3b>(pt.y-1, pt.x+1)[0] = color[0];
            srcMat.at<Vec3b>(pt.y-1, pt.x+1)[1] = color[1];
            srcMat.at<Vec3b>(pt.y-1, pt.x+1)[2] = color[2];
        }
    }

    imwrite("srcimage/outline.jpg", srcMat);
}
/*
* 函数功能：
* 通过凸点个数来判别种子，输入二值图，输出因种子粘连而产生的凸缺陷个数
*/
int SeedProcess::judgeHowMany(Mat& partBinaryMat)
{
    Mat tempMat;
    partBinaryMat.copyTo(tempMat);
    vector<vector<Point>> contours = getContours(tempMat);
    //vector<vector<Point>> contours = getContours(partBinaryMat);
    vector<vector<Point>>::const_iterator itc = contours.begin();
    for (; itc != contours.end();)
    {
        double areas = contourArea(Mat(*itc));
        if (areas < 21)
        {
            itc = contours.erase(itc);
        }
        else
        {
            ++itc;
        }
    }
    //vector<vector<Point> >hull(contours.size());
    vector<vector<int>> hullsI(contours.size());
    vector<vector<Vec4i>> defects(contours.size());         //轮廓凸缺陷，4维整型
    //convexHull(Mat(contours[0]), hull[0], false);
    convexHull(Mat(contours[0]), hullsI[0], false);
    convexityDefects(Mat(contours[0]), hullsI[0], defects[0]);//寻找轮廓凸缺陷，输出到defects[0]中，4维整型数据前3个分别对应contour[0]中的下标索引
    //分别为：凸缺陷在轮廓上起始点、结束点、离轮廓凸包最远点的下标索引、256倍离凸包最远距离

    vector<Vec4i>::iterator d = defects[0].begin();       //迭代器指向第一个凸缺陷
    vector<Point> points;
    while (d != defects[0].end())    //遍历凸缺陷
    {
        Vec4i& v = (*d);
        int faridx = v[2];
        Point ptFar(contours[0][faridx]);        //凸缺陷中离轮廓凸包最远点
        int depth = v[3] / 256;                  //最远距离
        if (depth > 4 && depth < 80)			 //通过最远距离，判别凸缺陷是否是因为多个种子粘连而产生
        {
            points.push_back(ptFar);		     //将判别后，因粘连而产生的凸缺陷相关点添加到points中
        }
        d++;
    }
    return points.size();        //返回因种子粘连而产生的凸缺陷个数
}
/*
 * @函数功能:int转string
 * @输入参数:无
 * @输出参数:无
 * @注意事项:
 *      无
 */
void int2str(const int &int_temp,string &string_temp)
{
    stringstream stream;
    stream<<int_temp;
    string_temp=stream.str();   //此处也可以用 stream>>string_temp
}
