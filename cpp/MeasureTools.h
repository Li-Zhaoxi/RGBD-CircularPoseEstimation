#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// convert ellipse shape parms to equation parms
// src: elpshape [x0,y0,a,b,theta]
// dst: outparms [a1,a2,a3,a4,a5,a6]
void ELPShape2Equation(double *elpshape, double *outparms);

// 计算椭圆一般方程当y值已知时，计算出的两个交点，如果不存在则为空集
void CalculateRangeAtY(double *elpparm, double y, double *x1, double *x2);



// 计算椭圆的重合度
void CalculateOverlap(double *elp1, double *elp2, double *ration, std::vector<double> *overlapdot);

// 给定一个椭圆，计算其y的取值范围
void CalculateRangeOfY(double *elpshape, double *x_min, double *x_max, double *y_min, double *y_max);

// 更快速的计算椭圆overlap的一种方法
void fasterCalculateOverlap(double *elp1, double *elp2, double *ration, std::vector<double> *overlapdot);

// 单椭圆计算位姿
void GetCirclePos(double* _C, double* _K, double Radius, double* _X1, double* _X2, double* _N1, double* _N2);

// 拟合圆
void fitCircle(int N, double*_pts, double *_shape_parms);
void fitEllipse(int N, double*_pts, double *_shape_parms);