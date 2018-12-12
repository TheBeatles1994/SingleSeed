#if 0
#include "algo.h"

int main(int argc, char *argv[])
{
    testSeed("srcimage/"+ _myfilename + ".jpg");

    cout<<"Finished!"<<endl;

    return 0;
}
#else
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;
using namespace std;

int main()
{
    //读入图像并进行灰度处理
    Mat srcImage1 = imread("1.jpg",0);
    Mat srcImage2 = imread("2.jpg",0);
    //cout<<"img1矩阵大小："<<img1.size()<<endl;
    //cout<<"img2矩阵大小："<<img2.size()<<endl;
    Mat tmpImage1,img1;
    Mat tmpImage2,img2;
    //对原图像进行两次缩放处理
    pyrDown(srcImage1,tmpImage1,Size(srcImage1.cols/2,srcImage1.rows/2));
    pyrDown(tmpImage1,img1,Size(tmpImage1.cols/2,tmpImage1.rows/2));
    pyrDown(srcImage2,tmpImage2,Size(srcImage2.cols/2,srcImage2.rows/2));
    pyrDown(tmpImage2,img2,Size(tmpImage2.cols/2,tmpImage2.rows/2));
    imshow("Src1",img1);
    imshow("Src2",img2);
    //imwrite("12.jpg",img1);
    //imwrite("13.jpg",img2);
    //第一步，用SURF算子检测关键点；
    int minHessian=400;
    SurfFeatureDetector detector(minHessian);
    std::vector<KeyPoint> m_LeftKey,m_RightKey;//构造2个专门由点组成的点向量用来存储特征点
    detector.detect(img1,m_LeftKey);//将img1图像中检测到的特征点存储起来放在m_LeftKey中
    detector.detect(img2,m_RightKey);//同理
    cout<<"图像1特征点的个数："<<m_LeftKey.size()<<endl;
    cout<<"图像2特征点的个数："<<m_RightKey.size()<<endl;
    //计算特征向量
    SurfDescriptorExtractor extractor;//定义描述子对象
    cv::Mat descriptors1, descriptors2;//存放特征向量的矩阵
    extractor.compute(img1,m_LeftKey,descriptors1);
    extractor.compute(img2,m_RightKey,descriptors2);
    cout<<"图像1特征描述矩阵大小："<<descriptors1.size()
       <<"，特征向量个数："<<descriptors1.rows<<"，维数："<<descriptors1.cols<<endl;
    cout<<"图像2特征描述矩阵大小："<<descriptors2.size()
       <<"，特征向量个数："<<descriptors2.rows<<"，维数："<<descriptors2.cols<<endl;
    //画出特征点
    Mat img_m_LeftKey,img_m_RightKey;
    drawKeypoints(img1,m_LeftKey,img_m_LeftKey,Scalar::all(-1),0);  //cvScalar(255,0,0)画的圈圈是蓝色，对应于特征点的颜色,DrawMatchesFlags::DRAW_RICH_KEYPOINTS表示关键点上圆圈的尺寸与特征的尺度成正比，对应于0，是“标志位”的意思
    drawKeypoints(img2,m_RightKey,img_m_RightKey,Scalar::all(-1),0);
    imshow("Keysrc1",img_m_LeftKey);
    imshow("Keysrc2",img_m_RightKey);
    imwrite("图像1的特征点.jpg",img_m_LeftKey);
    imwrite("图像2的特征点.jpg",img_m_RightKey);
    //匹配两幅图像的描述子
    //用burte force进行匹配特征向量
    BruteForceMatcher<L2<float>>matcher;//定义一个burte force matcher对象
    vector<DMatch> matches;//定义数据类型为matches的vector容器
    matcher.match( descriptors1, descriptors2, matches );//匹配两个图像的特征矩阵
    cout<<"Match个数："<<matches.size()<<endl;
    //计算匹配结果中距离的最大和最小值
    //距离是指两个特征向量间的欧式距离，表明两个特征的差异，值越小表明两个特征点越接近
    double max_dist = 0;
    double min_dist = 100;
    for(int i=0; i<matches.size(); i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    cout<<"最大距离："<<max_dist<<endl;
    cout<<"最小距离："<<min_dist<<endl;

    //筛选出较好的匹配点
    vector<DMatch> goodMatches;
    for(int i=0; i<matches.size(); i++)
    {
        if(matches[i].distance < 0.2 * max_dist)
        {
            goodMatches.push_back(matches[i]);
        }
    }
    cout<<"goodMatch个数："<<goodMatches.size()<<endl;
    //画出匹配结果
    Mat img_matches;
    //红色连接的是匹配的特征点对，绿色是未匹配的特征点
    drawMatches(img1,m_LeftKey,img2,m_RightKey,goodMatches,img_matches,
                Scalar::all(-1)/*CV_RGB(255,0,0)*/,CV_RGB(0,255,0),Mat(),2);
    imshow("MatchSURF",img_matches);
    waitKey(0);
    //RANSAC匹配过程
    vector<DMatch> m_Matches=goodMatches;
    //cout<<"m_Matches="<<m_Matches.size()<<endl;
    // 分配空间
    int ptCount = (int)m_Matches.size();
    //cout<<"m_Matches="<<ptCount<<endl;
    Mat p1(ptCount, 2, CV_32F);
    Mat p2(ptCount, 2, CV_32F);
    //cout<<"p1="<<p1<<endl;
    // 把Keypoint转换为Mat
    Point2f pt;
    for (int i=0; i<ptCount; i++)
    {
        pt = m_LeftKey[m_Matches[i].queryIdx].pt;
        p1.at<float>(i, 0) = pt.x;
        p1.at<float>(i, 1) = pt.y;
        pt = m_RightKey[m_Matches[i].trainIdx].pt;
        p2.at<float>(i, 0) = pt.x;
        p2.at<float>(i, 1) = pt.y;
    }
    //cout<<"p1="<<p1<<endl;//图1的匹配点坐标
    //cout<<"p2="<<p2<<endl;//图2的匹配点坐标
    // 用RANSAC方法计算F(基础矩阵)
    Mat m_Fundamental;
    vector<uchar> m_RANSACStatus;       // 这个变量用于存储RANSAC后每个点的状态
    findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
    // 计算内点个数
    int OutlinerCount = 0;
    for (int i=0; i<ptCount; i++)
    {
        if (m_RANSACStatus[i] == 0)    // 状态为0表示外点
        {
            OutlinerCount++;
        }
    }
    int InlinerCount = ptCount - OutlinerCount;   // 计算内点
    cout<<"内点数为："<<InlinerCount<<endl;
    cout<<"外点数为："<<OutlinerCount<<endl;
    // 这三个变量用于保存内点和匹配关系
    vector<Point2f> m_LeftInlier;
    vector<Point2f> m_RightInlier;
    vector<DMatch> m_InlierMatches;
    m_InlierMatches.resize(InlinerCount);
    m_LeftInlier.resize(InlinerCount);
    m_RightInlier.resize(InlinerCount);
    InlinerCount=0;
    float inlier_minRx=img1.cols;        //用于存储内点中右图最小横坐标，以便后续融合
    //cout<<"inlier="<<inlier_minRx<<endl;
    for (int i=0; i<ptCount; i++)
    {
        if (m_RANSACStatus[i] != 0)
        {
            m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
            m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
            m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
            m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
            m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
            m_InlierMatches[InlinerCount].trainIdx = InlinerCount;

            if(m_RightInlier[InlinerCount].x<inlier_minRx) inlier_minRx=m_RightInlier[InlinerCount].x;   //存储内点中右图最小横坐标
            InlinerCount++;
        }
    }
    //cout<<"inlier="<<inlier_minRx<<endl;
    // 把内点转换为drawMatches可以使用的格式
    vector<KeyPoint> key1(InlinerCount);
    vector<KeyPoint> key2(InlinerCount);
    KeyPoint::convert(m_LeftInlier, key1);
    KeyPoint::convert(m_RightInlier, key2);
    // 显示计算F过后的内点匹配
    Mat OutImage;
    drawMatches(img1, key1, img2, key2, m_InlierMatches, OutImage);
    imshow("RANSAC match features", OutImage);
    waitKey(0);
    //矩阵H用以存储RANSAC得到的单应矩阵
    Mat H = findHomography( m_LeftInlier, m_RightInlier, RANSAC );
    //存储左图四角，及其变换到右图位置
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0,0); obj_corners[1] = Point( img1.cols, 0 );
    obj_corners[2] = Point( img1.cols, img1.rows ); obj_corners[3] = Point( 0, img1.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);
    //画出变换后图像位置
    Point2f offset( (float)img1.cols, 0);
    line( OutImage, scene_corners[0]+offset, scene_corners[1]+offset, Scalar( 0, 255, 0), 4 );
    line( OutImage, scene_corners[1]+offset, scene_corners[2]+offset, Scalar( 0, 255, 0), 4 );
    line( OutImage, scene_corners[2]+offset, scene_corners[3]+offset, Scalar( 0, 255, 0), 4 );
    line( OutImage, scene_corners[3]+offset, scene_corners[0]+offset, Scalar( 0, 255, 0), 4 );
    //imshow( "Good Matches & Object detection", OutImage );
    int drift = scene_corners[1].x;
    //储存偏移量
    cout<<"scene="<<scene_corners<<endl;
    cout<<"scene0="<<scene_corners[0].x<<endl;
    cout<<"scene1="<<scene_corners[1].x<<endl;
    cout<<"scene2="<<scene_corners[2].x<<endl;
    cout<<"scene3="<<scene_corners[3].x<<endl;
    //新建一个矩阵存储配准后四角的位置
    int width = int(max(abs(scene_corners[1].x), abs(scene_corners[2].x)));
    int height= img1.rows;                                                                  //或者：int height = int(max(abs(scene_corners[2].y), abs(scene_corners[3].y)));
    float origin_x=0,origin_y=0;
    if(scene_corners[0].x<0) {
        if (scene_corners[3].x<0) origin_x+=min(scene_corners[0].x,scene_corners[3].x);
        else origin_x+=scene_corners[0].x;}
    width-=int(origin_x);
    if(scene_corners[0].y<0) {
        if (scene_corners[1].y) origin_y+=min(scene_corners[0].y,scene_corners[1].y);
        else origin_y+=scene_corners[0].y;}
    //可选：height-=int(origin_y);
    Mat imageturn=Mat::zeros(width,height,img1.type());
    cout<<"width: "<<width<<endl;
    cout<<"height: "<<height<<endl;
    cout<<"img1.type(): "<<img1.type()<<endl;
    //获取新的变换矩阵，使图像完整显示
    for (int i=0;i<4;i++) {scene_corners[i].x -= origin_x; } 	//可选：scene_corners[i].y -= (float)origin_y; }
    Mat H1=getPerspectiveTransform(obj_corners, scene_corners);
    //进行图像变换，显示效果
    warpPerspective(img1,imageturn,H1,Size(width,height));
    imshow("image_Perspective", imageturn);
    waitKey(0);
    cout<<"origin_x="<<origin_x<<endl;
    cout<<"origin_y="<<origin_y<<endl;
    cout<<"width="<<width<<endl;
    cout<<"img1.width="<<img1.cols<<endl;
    cout<<"height="<<height<<endl;
    cout<<"inlier_minRx="<<inlier_minRx<<endl;
    cout<<"scene_corners= "<<scene_corners<<endl;
    cout<<"单应矩阵H="<<H<<endl;
    cout<<"变换矩阵H1="<<H1<<endl;
    //图像融合
    int width_ol=width-int(inlier_minRx-origin_x);
    int start_x=int(inlier_minRx-origin_x);
    uchar* ptr=imageturn.data;
    double alpha=0, beta=1;
    for (int row=0;row<height;row++) {
        ptr=imageturn.data+row*imageturn.step+(start_x)*imageturn.elemSize();
        for(int col=0;col<width_ol;col++)
        {
            uchar* ptr_c1=ptr+imageturn.elemSize1();  uchar*  ptr_c2=ptr_c1+imageturn.elemSize1();
            uchar* ptr2=img2.data+row*img2.step+(col+int(inlier_minRx))*img2.elemSize();
            uchar* ptr2_c1=ptr2+img2.elemSize1();  uchar* ptr2_c2=ptr2_c1+img2.elemSize1();
            alpha=double(col)/double(width_ol); beta=1-alpha;
            if (*ptr==0&&*ptr_c1==0&&*ptr_c2==0) {
                *ptr=(*ptr2);
                *ptr_c1=(*ptr2_c1);
                *ptr_c2=(*ptr2_c2);
            }
            *ptr=(*ptr)*beta+(*ptr2)*alpha;
            *ptr_c1=(*ptr_c1)*beta+(*ptr2_c1)*alpha;
            *ptr_c2=(*ptr_c2)*beta+(*ptr2_c2)*alpha;
            ptr+=imageturn.elemSize();
        }
    }
    imshow("image_overlap", imageturn);
    waitKey(0);
    Mat img_result=Mat::zeros(height,width+img2.cols-drift,img1.type());
    uchar* ptr_r=imageturn.data;
    for (int row=0;row<height;row++) {
        ptr_r=img_result.data+row*img_result.step;
        for(int col=0;col<imageturn.cols;col++)
        {
            uchar* ptr_rc1=ptr_r+imageturn.elemSize1();  uchar*  ptr_rc2=ptr_rc1+imageturn.elemSize1();
            uchar* ptr=imageturn.data+row*imageturn.step+col*imageturn.elemSize();
            uchar* ptr_c1=ptr+imageturn.elemSize1();  uchar*  ptr_c2=ptr_c1+imageturn.elemSize1();
            *ptr_r=*ptr;
            *ptr_rc1=*ptr_c1;
            *ptr_rc2=*ptr_c2;
            ptr_r+=img_result.elemSize();
        }
        ptr_r=img_result.data+row*img_result.step+imageturn.cols*img_result.elemSize();
        for(int col=imageturn.cols;col<img_result.cols;col++)
        {
            uchar* ptr_rc1=ptr_r+imageturn.elemSize1();  uchar*  ptr_rc2=ptr_rc1+imageturn.elemSize1();
            uchar* ptr2=img2.data+row*img2.step+(col-imageturn.cols+drift)*img2.elemSize();
            uchar* ptr2_c1=ptr2+img2.elemSize1();  uchar* ptr2_c2=ptr2_c1+img2.elemSize1();
            *ptr_r=*ptr2;
            *ptr_rc1=*ptr2_c1;
            *ptr_rc2=*ptr2_c2;
            ptr_r+=img_result.elemSize();
        }
    }
    imshow("image_result", img_result);
    //imwrite("14.jpg",img_result);
    waitKey(0);
    return 0;
}


#endif
