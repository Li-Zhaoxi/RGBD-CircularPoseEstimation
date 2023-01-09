#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// convert ellipse shape parms to equation parms
// src: elpshape [x0,y0,a,b,theta]
// dst: outparms [a1,a2,a3,a4,a5,a6]
void ELPShape2Equation(double *elpshape, double *outparms);

// ������Բһ�㷽�̵�yֵ��֪ʱ����������������㣬�����������Ϊ�ռ�
void CalculateRangeAtY(double *elpparm, double y, double *x1, double *x2);



// ������Բ���غ϶�
void CalculateOverlap(double *elp1, double *elp2, double *ration, std::vector<double> *overlapdot);

// ����һ����Բ��������y��ȡֵ��Χ
void CalculateRangeOfY(double *elpshape, double *x_min, double *x_max, double *y_min, double *y_max);

// �����ٵļ�����Բoverlap��һ�ַ���
void fasterCalculateOverlap(double *elp1, double *elp2, double *ration, std::vector<double> *overlapdot);

// ����Բ����λ��
void GetCirclePos(double* _C, double* _K, double Radius, double* _X1, double* _X2, double* _N1, double* _N2);

// ���Բ
void fitCircle(int N, double*_pts, double *_shape_parms);
void fitEllipse(int N, double*_pts, double *_shape_parms);