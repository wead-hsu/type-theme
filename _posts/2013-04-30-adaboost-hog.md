---
layout: default
title: 基于HOG特征的Adaboost行人检测
categories: machine-learning
---

本文将介绍一种基于HOG特征的行人检测方法。提取子图的特征并用于训练简单分类器，然后根据adaboost调整样本权重，得到强分类器。


- 主要方法
	* 方向梯度直方图（Histogramof Oriented Gradient, HOG）特征是一种在计算机视觉和图像处理中用来进行物体检测的特征描述子。它通过计算和统计图像局部区域的梯度方向直方图来构成特征。基本知识可以参考[博客](http://blog.csdn.net/zouxy09/article/details/7929348)
	* Adaboost的基础知识可以参考书籍：统计学习方法，第八章-提升方法adaboost。

这里利用HOG来训练Adaboost行人检测。在Haar-Adaboost算法中，弱分类器仅对一维分类。但是在Hog特征中，特征是每个block的串联。如果仅对一维分类（一个cell的其中一个方向的权值），就不能有效利用block的归一化效果。所以我们使用logistic弱分类器对每个block进行分类（实验中，每个block包含4个cell，每个cell有9个bin，即36维特征）。

- 本实验需要注意的地方
	* adaboost误差率需要计算权重 
	* logistic回归需要使用带权重的logistic分类器 

- 实验结果
	* 训练集: 500/500;测试集: 19/22（200个弱分类器）

- 实验代码

```cpp
/***********************************************************/
/** Copyright by Weidi Xu, S.C.U.T in Guangzhou, Guangdong**/
/***********************************************************/

#include <opencv2\opencv.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <ctime>

using std::clock_t;
using std::clock;
using namespace cv;

//const parameters for image
const int NUM_NEGIMAGE = 1000;
const int NUM_POSIMAGE = 500;
const int NUM_IMAGE = 1500;
const int NUM_TESTIMAGE = 22;
const int MAX_DIMENSION = 3781;
const int IMAGE_ROWS = 128;
const int IMAGE_COLS = 64;
const int CELLSIZE = 8;
const int BLOCKSIZE = 16;
const int MOVELENGTH = 8;
const int BINSIZE = 9;
const double PI = 2*acos(0.0);
const double eps = 1e-8;

//mediate parameter
const int NUM_BLOCK_ROWS = (IMAGE_ROWS-BLOCKSIZE)/MOVELENGTH+1;
const int NUM_BLOCK_COLS = (IMAGE_COLS-BLOCKSIZE)/MOVELENGTH+1;
const int NUM_BLOCK_FEATURES = (BLOCKSIZE/CELLSIZE)*(BLOCKSIZE/CELLSIZE)*BINSIZE+1;//zero for theta[0]

//data from image
//since the features in the adaboost should contain a single block, it's better to define the feature of 3-dimension;
double features[NUM_IMAGE][NUM_BLOCK_ROWS][NUM_BLOCK_COLS][NUM_BLOCK_FEATURES];
double type[NUM_IMAGE]; //1 - pos, 0 - neg
double y[NUM_IMAGE]; //1 - pos, -1 - neg

//number of weak classifier(changing in experiment)
const int NUM_WEAKCLASSIFIER = 100;

//data for adaboost
double weight[NUM_IMAGE];

//logistic function(dimension is given by NUM_BLOCK_FEATURES(37 in this setting))
double logistic(double theta[], double x[])
{
    double ans = 0;
    for(int i = 0 ; i < NUM_BLOCK_FEATURES; i++)
    {
        ans += theta[i]*x[i];
    }
    return 1/(1+std::exp(-ans));
}

struct WeakClassifier
{
    double _theta[NUM_BLOCK_FEATURES]; //threshold classifier
    int _index_row;  //classify by the features in block[_block_row][_block_y]
    int _index_col;
    int _isreverse; //1 for (> := pos, < := neg); 0 for (< := pos, >:= neg)
    double _alpha; 
    double _error;
    void clear()
    {
        memset(_theta, 0.0, NUM_BLOCK_FEATURES*sizeof(double));
        _alpha = 0.0;
        _error = 1;
        _index_row = -1;
        _index_col = -1;
        _isreverse = true;
    }

    //return 1 or -1
    int cal(double x[NUM_BLOCK_ROWS][NUM_BLOCK_COLS][NUM_BLOCK_FEATURES])
    {
        int ans = logistic(_theta, x[_index_row][_index_col]);
        if(ans > 0.5)
        {
            if(_isreverse)
                return -1;
            else
                return 1;
        }
        else
        {
            if(_isreverse)
                return 1;
            else
                return -1;
        }
    }

    void print()
    {
        //theta
        for(int i = 0 ; i < NUM_BLOCK_FEATURES; i++)
        printf("%lf ", _theta[i]);
        printf("\n");

        //int _index_row;
        printf("%d ",_index_row);

        //int _index_col;
        printf("%d ",_index_col);

        //int _isreverse; //1 for (> := pos, < := neg); 0 for (< := pos, >:= neg)
        printf("%d ",_isreverse);

        //double _alpha;
        printf("%lf ",_alpha);
        
        //double _error;
        printf("%lf \n",_error);
    }
}weakClassifier[NUM_WEAKCLASSIFIER];

//Util Function
double arc2angle(double arc)
{
    return arc/PI*180.0;
}

double angle2arc(double angle)
{
    return angle/180.0*PI;
}

void posfilename(int i, char* filename)
{
    sprintf(filename, "pos/pos (%d).png", i);
    return;
}

void negfilename(int i, char* filename)
{
    sprintf(filename, "neg/neg (%d).png", i);
    return;
}

void testfilename(int i, char* filename)
{
    sprintf(filename, "test_pos/test (%d).png", i);
    return ;
}

//I(x,y) = sqrt(I(x,y))
void normalizeImage(Mat& inputImage)
{
    // accept only char type matrices  
    CV_Assert(inputImage.depth() != sizeof(uchar));  
    int channels = inputImage.channels();  
    int nRows = inputImage.rows ;  
    int nCols = inputImage.cols* channels;  
    if (inputImage.isContinuous())  
    {  
        nCols *= nRows;  
        nRows = 1;  
    }  
    int i,j;  
    uchar* p;  
    for( i = 0; i < nRows; ++i)  
    {  
        p = inputImage.ptr<uchar>(i);  
        for ( j = 0; j < nCols; ++j)  
        {  
            p[j] = int(sqrt(p[j]*1.0));  
        }  
    }
    return;
}

//I(x,y) 第一维的梯度为xGradient
void calGredient(const Mat& inputImage, double xGradient[IMAGE_ROWS][IMAGE_COLS], double yGradient[IMAGE_ROWS][IMAGE_COLS])
{
    uchar* dataptr = inputImage.data;
    int nrows = inputImage.rows;
    int ncols = inputImage.cols;

    //cal xgradient
    for(int i = 1 ; i < nrows - 1; i++)
    {
        for(int j = 0 ; j < ncols; j++)
        {
            xGradient[i][j] = inputImage.at<uchar>(i+1,j) - inputImage.at<uchar>(i-1,j);
        }
    }

    //cal margin
    for(int i = 0 ; i < ncols; i++)
    {
        xGradient[0][i] = (inputImage.at<uchar>(1,i) - inputImage.at<uchar>(0,i))*2;
        xGradient[nrows-1][i] = (inputImage.at<uchar>(nrows-1,i) - inputImage.at<uchar>(nrows-2,i))*2;
    }

    //cal ygradient
    for(int i = 0 ; i < nrows ; i++)
    {
        for(int j = 1 ; j < ncols - 1; j++)
        {
            yGradient[i][j] = inputImage.at<uchar>(i,j+1) - inputImage.at<uchar>(i,j-1);
        }
    }

    //cal margin
    for(int i = 0 ; i < nrows; i++)
    {
        xGradient[i][0] = (inputImage.at<uchar>(i,1) - inputImage.at<uchar>(i,0))*2;
        xGradient[i][ncols-1] = (inputImage.at<uchar>(i,ncols-1) - inputImage.at<uchar>(i,ncols-2))*2;
    }
}

//cal the HogFeatures by block
void calHogFeatures(Mat& inputImage, double outputFeature[NUM_BLOCK_ROWS][NUM_BLOCK_COLS][NUM_BLOCK_FEATURES])
{
    int nrows = inputImage.rows;
    int ncols = inputImage.cols;
    int type = inputImage.type();

    if(nrows != IMAGE_ROWS || ncols != IMAGE_COLS)
        abort();

    //cal x,yGradient
    double xGradient[IMAGE_ROWS][IMAGE_COLS];
    double yGradient[IMAGE_ROWS][IMAGE_COLS];
    calGredient(inputImage, xGradient, yGradient);

    //computation median
    double gradient[IMAGE_ROWS][IMAGE_COLS];
    double direction[IMAGE_ROWS][IMAGE_COLS];

    for(int i = 0 ; i < nrows; i++)
    {
        for(int j = 0 ; j < ncols; j++)
        {
            double gx = xGradient[i][j];
            double gy = yGradient[i][j];
            gradient[i][j] = sqrt(gx*gx + gy*gy);
            direction[i][j] = arc2angle(atan2(gy, gx));
        }
    }

    //compute cellinfo 8*8
    double cellinfo[IMAGE_ROWS/CELLSIZE][IMAGE_COLS/CELLSIZE][BINSIZE];
    memset(cellinfo, 0, sizeof(cellinfo));

    for(int i = 0; i < IMAGE_ROWS/CELLSIZE; i++)
    {
        for(int j = 0 ; j < IMAGE_COLS/CELLSIZE; j++)
        {
            double* cell = cellinfo[i][j];
            
            //cal single cellinfo of 8*8
            for(int ci = 0 ; ci < CELLSIZE; ci++)
            {
                for(int cj = 0; cj < CELLSIZE; cj++)
                {
                    //find org pix;
                    int px = i*CELLSIZE + ci;
                    int py = j*CELLSIZE + cj;

                    int binindex = int((direction[px][py]+180.0)/(360.0/BINSIZE));
                    //handle bound 
                    if(fabs(direction[px][py]-180) < eps)
                    {
                        binindex = BINSIZE-1;
                    }
                    if(fabs(direction[px][py]+180) < eps)
                    {
                        binindex = 0;
                    }
                    if(binindex < 0 || binindex >= BINSIZE)
                    {
                        printf("Wrong binindex: %d %lf %lf %lf", binindex, xGradient[px][py], yGradient[px][py], direction[px][py]);
                        abort();
                    }

                    cell[binindex] += gradient[px][py];
                }
            }
        }
    }

    /*double blockinfo[(IMAGE_ROWS-BLOCKSIZE)/MOVELENGTH+1][(IMAGE_COLS-BLOCKSIZE)/MOVELENGTH+1][(BLOCKSIZE/CELLSIZE)*(BLOCKSIZE/CELLSIZE)*BINSIZE];*/

    if(MOVELENGTH%CELLSIZE != 0)
    {
        printf("MOVELENGTH%CELLSIZE != 0");
        abort();
    }

    //cal blockinfo
    for(int i = 0 ; i < (IMAGE_ROWS-BLOCKSIZE)/MOVELENGTH + 1; i++)
    {
        for(int j = 0 ; j < (IMAGE_COLS-BLOCKSIZE)/MOVELENGTH + 1; j++)
        {
            int bfindex = 0; outputFeature[i][j][bfindex++] = 1;

            //cal the position of this block
            for(int c1 = 0; c1 < BLOCKSIZE/CELLSIZE; c1++)
            {
                for(int c2 = 0 ; c2 < BLOCKSIZE/CELLSIZE; c2++)
                {
                    //cal the index of cell
                    int cx = i*MOVELENGTH/CELLSIZE+c1;
                    int cy = j*MOVELENGTH/CELLSIZE+c2;

                    for(int binindex = 0 ; binindex < BINSIZE; binindex++)
                    {
                        outputFeature[i][j][bfindex++] = cellinfo[cx][cy][binindex];
                    }
                }
            }
        }
    }
    return;
}

//use global variables
void trainLogisticRegression(int block_row,int block_col, double theta[], double& errorrate, int& isreverse)
{
    double theta1[NUM_BLOCK_FEATURES], theta2[NUM_BLOCK_FEATURES];
    memset(theta1, 0, NUM_BLOCK_FEATURES*sizeof(double));
    memset(theta2, 0, NUM_BLOCK_FEATURES*sizeof(double));
    double errorrate1 = 0;
    double errorrate2 = 0;
    double rightnum1 = 0;
    double rightnum2 = 0;
    isreverse = 0;

    //cal parameter thetas
    for(int k = 0 ; k < 100000; k++)
    {
        int i = rand()%NUM_IMAGE;
        int j = rand()%NUM_BLOCK_FEATURES;
        theta1[j] = theta1[j] + weight[i]*0.01*(type[i] - logistic(theta1, features[i][block_row][block_col]))*features[i][block_row][block_col][j];
    }

    for(int i = 0 ; i < NUM_IMAGE; i++)
    {
        double tmp = logistic(theta1, features[i][block_row][block_col]);
        if(tmp > 0.5 && fabs(type[i] - 1) < eps)
            rightnum1 += 1.0*weight[i];
        if(tmp < 0.5 && fabs(type[i] - 0) < eps)
            rightnum1 += 1.0*weight[i];
    }
    errorrate1 = 1 - rightnum1;

    //calreverse
    for(int k = 0 ; k < 100000; k++)
    {
        int i = rand()%NUM_IMAGE;
        int j = rand()%NUM_BLOCK_FEATURES;
        theta2[j] = theta2[j] + weight[i]*0.01*(1- type[i] - logistic(theta2, features[i][block_row][block_col]))*features[i][block_row][block_col][j];
    }

    for(int i = 0 ; i < NUM_IMAGE; i++)
    {
        double tmp = logistic(theta2, features[i][block_row][block_col]);
        if(tmp > 0.5 && fabs(type[i] - 0) < eps)
            rightnum2 += 1.0*weight[i];
        if(tmp < 0.5 && fabs(type[i] - 1) < eps)
            rightnum2 += 1.0*weight[i];
    }
    errorrate2 = 1 - rightnum2;

    if(errorrate1 < errorrate2)
    {
        for(int i = 0 ; i < NUM_BLOCK_FEATURES; i++)
        {
            theta[i] = theta1[i];
        }
        isreverse = 0;
        errorrate = errorrate1 + eps;
    }
    else
    {
        for(int i = 0 ; i < NUM_BLOCK_FEATURES; i++)
        {
            theta[i] = theta2[i];
        }
        isreverse = 1;
        errorrate = errorrate2 + eps;
    }
    return;
}

WeakClassifier trainClassifier()
{
    WeakClassifier ansclassifier;
    double theta[NUM_BLOCK_FEATURES];
    double errorrate = 1;
    int isreverse = 0;
    double best_theta[NUM_BLOCK_FEATURES];
    double best_errorrate = 1;
    int best_row = -1;
    int best_col = -1;
    int best_isreverse = 0;

    //select best weak classifier
    for(int i = 0 ; i < NUM_BLOCK_ROWS; i++)
    {
        for(int j = 0 ; j < NUM_BLOCK_COLS; j++)
        {
            trainLogisticRegression(i,j,theta,errorrate, isreverse);
            
            if(errorrate < 0)
            {
                printf("Wrong errorrate < 0 : %lf", errorrate);
                abort();
            }

            if(errorrate < best_errorrate)
            {
                for(int tempi = 0 ; tempi < NUM_BLOCK_FEATURES; tempi++)
                {
                    best_theta[tempi] = theta[tempi];
                }
                best_errorrate = errorrate;
                best_row = i;
                best_col = j;
                best_isreverse = isreverse;
            }
        }
    }

    if(best_errorrate > 0.5)
    {
        printf("The best_errorrate is greater than 0.5.\n");
        abort();
    }

    //set parameters;
    ansclassifier._alpha = 1.0/2*std::log((1-best_errorrate)/best_errorrate);
    ansclassifier._error = best_errorrate;
    ansclassifier._index_col = best_col;
    ansclassifier._index_row = best_row;
    ansclassifier._isreverse = best_isreverse;
    for(int i = 0 ; i < NUM_BLOCK_FEATURES; i++) ansclassifier._theta[i] = best_theta[i];

    return ansclassifier;
}

int calByStrongClassifier(double x[NUM_BLOCK_ROWS][NUM_BLOCK_COLS][NUM_BLOCK_FEATURES])
{
    double ans = 0;
    for(int i = 0 ; i < NUM_WEAKCLASSIFIER; i++)
    {
        ans += weakClassifier[i]._alpha * weakClassifier[i].cal(x);
    }
    if(ans > 0)
        return 1;
    else
        return -1;
}



/*
size: 128*64;
type: CV_8UC1;
Block大小为18*18；
Cell大小为6*6；
Block在检测窗口中上下移动尺寸为6*6；
1个cell的梯度直方图化成9个bin；
滑动窗口在检测图片中滑动的尺寸为6*6；
*/

int main()
{
    char filename[100];
    IplImage* inputImage = NULL;
    clock_t timecount = clock();

    //load posimage
    for(int i = 0 ; i < NUM_POSIMAGE; i++)
    {
        posfilename(i+1 ,filename);
        
        //load grey image: set the parameter to 0;
        inputImage = cvLoadImage(filename, 0);
        
        
        //cal features;
        Mat inputMat(inputImage);
        calHogFeatures(inputMat, features[i]);
        type[i] = 1;
        y[i] = 1;
        //printf("%d \n", inputMat.cols);

        //release memory
        inputMat.release();
        cvReleaseImage(&inputImage);
        inputImage = NULL;
    }

    printf("The feature process of pos-image have done in %d second.\n", (clock()-timecount)/1000);
    timecount = clock();

    //load neg images
    for(int i = 0; i < NUM_NEGIMAGE; i++)
    {
        negfilename(i+1, filename);

        //load grey image: set the parameter to 0;
        inputImage = cvLoadImage(filename, 0);
        type[NUM_POSIMAGE+i] = 0;
        y[NUM_POSIMAGE+i] = -1;

        Mat inputMat(inputImage);
        calHogFeatures(inputMat, features[NUM_POSIMAGE+i]);
        
        //release memory
        inputMat.release();
        cvReleaseImage(&inputImage);
        inputImage = NULL;
    }

    printf("The feature process of neg-image have done in %d second.\n", (clock()-timecount)/1000);
    timecount = clock();

    //init weight array
    for(int i = 0 ; i < NUM_IMAGE; i++)
    {
        weight[i] = 1.0/NUM_IMAGE;
    }

    //freopen
    freopen("HOG_CLASSIFIER.txt", "w", stdout);

    //print number of weakclassifiers;
    printf("%d\n", NUM_WEAKCLASSIFIER);

    //adaboost framework
    for(int classifierindex = 0 ; classifierindex < NUM_WEAKCLASSIFIER; classifierindex++)
    {
        weakClassifier[classifierindex] = trainClassifier();

        double error = weakClassifier[classifierindex]._error;
        double alpha = weakClassifier[classifierindex]._alpha;

        //printf("%d classifier: %lf ====\n",classifierindex, error);
        //printf("_index_row %d _index_col %d\n", weakClassifier[classifierindex]._index_row, weakClassifier[classifierindex]._index_col);

        double identitysum = 0;
        for(int sampleindex = 0 ; sampleindex < NUM_IMAGE; sampleindex++)
        {
            weight[sampleindex] *= std::exp(-alpha*y[sampleindex]*weakClassifier[classifierindex].cal(features[sampleindex]));
            identitysum += weight[sampleindex];
        }

        //reweight
        for(int sampleindex = 0 ; sampleindex < NUM_IMAGE; sampleindex++)
        {
            weight[sampleindex] /= identitysum;
        }

        weakClassifier[classifierindex].print();
    }

    freopen("CON", "w", stdout);
    int rightnum = 0;
    for(int testindex = 0 ;testindex < NUM_TESTIMAGE; testindex ++)
    {
        //posfilename(testindex+1, filename);
        testfilename(testindex+1, filename);
        inputImage = cvLoadImage(filename, 0);

        double testfeatures[NUM_BLOCK_ROWS][NUM_BLOCK_COLS][NUM_BLOCK_FEATURES];
        memset(testfeatures, 0, sizeof(testfeatures));

        Mat inputMat(inputImage);
        calHogFeatures(inputMat, testfeatures);

        if(calByStrongClassifier(testfeatures) == 1)
        {
            rightnum++;
            //printf("Yes\n");
        }
        else
            //printf("No\n");

        inputMat.release();
    }
    printf("Accuracy: %d\n", rightnum);
}

//测试数据是网上流行的128*64灰度行人图像数据。
```
 
