#include "MeasureTools.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define PIXEL_SCALE 0.1
#define PIXEL_SCALE_ELLIPSE_IMAGE 50.0

void GetPointMat(double x, double y, double scale, double *nodeMat)
{
    x /= scale;
    y /= scale;

    nodeMat[0] = pow(x, 4);
	nodeMat[1] = 2 * pow(x, 3)*y;
	nodeMat[2] = x*x*y*y;
	nodeMat[3] = 2 * pow(x, 3);
	nodeMat[4] = 2 * x*x*y;
	nodeMat[5] = x*x;
	nodeMat[6] = 2 * x*y*y*y;
	nodeMat[7] = 4 * x*y*y;
	nodeMat[8] = 2 * x*y;
	nodeMat[9] = pow(y, 4);
	nodeMat[10] = 2 * pow(y, 3);
	nodeMat[11] = y*y;
	nodeMat[12] = 2 * x;
	nodeMat[13] = 2 * y;
	nodeMat[14] = 1;
}

void fitCircle(double *data, double &error, cv::RotatedRect &res, double scale)
{
    double _buf_fit_data[36];
    int dot_num = data[14];
    cv::Point2d cen_dot(data[12] / (2 * dot_num), data[13] / (2 * dot_num));//the center of the fitting data.

    _buf_fit_data[0] = data[0] - 2 * cen_dot.x*data[3] + 6 * cen_dot.x*cen_dot.x*data[5] - 3 * dot_num*pow(cen_dot.x, 4);
    _buf_fit_data[1] = data[1] - 3 * cen_dot.x*data[4] + 3 * cen_dot.x*cen_dot.x*data[8] - cen_dot.y*data[3] + cen_dot.x*cen_dot.y * 6 * data[5] - 6 * dot_num*cen_dot.y*pow(cen_dot.x, 3);
    _buf_fit_data[2] = data[2] - cen_dot.y*data[4] + cen_dot.y*cen_dot.y*data[5] - cen_dot.x*data[7] / 2 + cen_dot.x*cen_dot.y*data[8] * 2 + cen_dot.x*cen_dot.x*data[11] - 3 * dot_num*pow(cen_dot.x*cen_dot.y, 2);
    _buf_fit_data[3] = data[3] - 6 * cen_dot.x*data[5] + 4 * dot_num*pow(cen_dot.x, 3);
    _buf_fit_data[4] = data[4] - cen_dot.x*data[8] * 2 - 2 * cen_dot.y*data[5] + 4 * dot_num*cen_dot.x*cen_dot.x*cen_dot.y;
    _buf_fit_data[5] = data[5] - dot_num*cen_dot.x*cen_dot.x;

    _buf_fit_data[1 * 6 + 1] = 4 * _buf_fit_data[2];
    _buf_fit_data[1 * 6 + 2] = data[6] - 1.5 * cen_dot.y*data[7] + 3 * cen_dot.y*cen_dot.y*data[8] - cen_dot.x*data[10] + 6 * cen_dot.x*cen_dot.y*data[11] - 6 * dot_num*cen_dot.x*pow(cen_dot.y, 3);
    _buf_fit_data[1 * 6 + 3] = 2 * _buf_fit_data[4];
    _buf_fit_data[1 * 6 + 4] = data[7] - 4 * cen_dot.y*data[8] - 4 * cen_dot.x*data[11] + 8 * dot_num*cen_dot.x*cen_dot.y*cen_dot.y;
    _buf_fit_data[1 * 6 + 5] = data[8] - 2 * dot_num*cen_dot.x*cen_dot.y;

    _buf_fit_data[2 * 6 + 2] = data[9] - 2 * cen_dot.y*data[10] + 6 * cen_dot.y*cen_dot.y*data[11] - 3 * dot_num*pow(cen_dot.y, 4);
    _buf_fit_data[2 * 6 + 3] = _buf_fit_data[1 * 6 + 4] / 2;
    _buf_fit_data[2 * 6 + 4] = data[10] - 6 * cen_dot.y*data[11] + 4 * dot_num*pow(cen_dot.y, 3);
    _buf_fit_data[2 * 6 + 5] = data[11] - dot_num*cen_dot.y*cen_dot.y;

    _buf_fit_data[3 * 6 + 3] = 4 * _buf_fit_data[5];
    _buf_fit_data[3 * 6 + 4] = 2 * _buf_fit_data[1 * 6 + 5];
    _buf_fit_data[3 * 6 + 5] = 0;

    _buf_fit_data[4 * 6 + 4] = 4 * _buf_fit_data[2 * 6 + 5];
    _buf_fit_data[4 * 6 + 5] = 0;

    _buf_fit_data[5 * 6 + 5] = dot_num;
    for (int i = 0; i < 6; i++)
    {
        for (int j = i + 1; j < 6; j++)
        {
            _buf_fit_data[j * 6 + i] = _buf_fit_data[i * 6 + j];
        }
    }

    double _buffer_circle_data[16];
    _buffer_circle_data[0] = _buf_fit_data[0] + 2 * _buf_fit_data[2] + _buf_fit_data[2 * 6 + 2];
    _buffer_circle_data[1] = _buf_fit_data[3] + 2 * _buf_fit_data[3 * 6 + 2];
    _buffer_circle_data[2] = _buf_fit_data[4] + _buf_fit_data[2 * 6 + 4];
    _buffer_circle_data[3] = _buf_fit_data[5] + _buf_fit_data[2 * 6 + 5];
    _buffer_circle_data[1 * 4 + 1] = _buf_fit_data[ 3 * 6 + 3];
    _buffer_circle_data[1 * 4 + 2] = _buf_fit_data[ 3 * 6 + 4];
    _buffer_circle_data[1 * 4 + 3] = _buf_fit_data[ 3 * 6 + 5];
    _buffer_circle_data[2 * 4 + 2] = _buf_fit_data[ 4 * 6 + 4];
    _buffer_circle_data[2 * 4 + 3] = _buf_fit_data[ 4 * 6 + 5];
    _buffer_circle_data[3 * 4 + 3] = dot_num;
    for (int i = 0; i < 4; i++)
    {
        for (int j = i + 1; j < 4; j++)
        {
            _buffer_circle_data[j * 4 + i] = _buffer_circle_data[i * 4 + j];
        }
    }
    cv::Mat DLS_C = cv::Mat::zeros(4,4,CV_64FC1), buf_circle_data(4, 4, CV_64FC1, _buffer_circle_data);
    DLS_C.at<double>(0, 3) = -0.5;
    DLS_C.at<double>(3, 0) = -0.5;
    DLS_C.at<double>(1, 1) = 1;
    DLS_C.at<double>(2, 2) = 1;

    cv::Mat dls_D, dls_V, dls_D2, dls_V2, dls_X;

    cv::eigen(buf_circle_data, dls_D, dls_V);
    double *_dls_D = (double*)dls_D.data;
    if (_dls_D[3] > 1e-10)
    {
        for (int i = 0; i < 4; i++)
            dls_V.row(i) = dls_V.row(i) / sqrt(fabs(_dls_D[i]));
        buf_circle_data = dls_V*DLS_C*dls_V.t();
        cv::eigen(buf_circle_data, dls_D2, dls_V2);
        //		cout << dls_D2 << '\n' << dls_V2 << endl;
        double *_dls_D2 = (double*)dls_D2.data;
        if (_dls_D2[0] > 0)
        {
            error = scale*scale / _dls_D2[0] / dot_num;
            dls_X = dls_V2.row(0)*dls_V;
        }
        else
        {
            error = -1;
            return;
        }
    }
    else
        dls_X = dls_V.row(3);


    double *_dls_X = (double*)dls_X.data;
    double dls_k = _dls_X[1] * _dls_X[1] + _dls_X[2] * _dls_X[2] - _dls_X[0] * _dls_X[3];
    dls_X = dls_X / sqrt(abs(dls_k));



    res.center.x = (-_dls_X[1]/_dls_X[0] + cen_dot.x)*scale;
    res.center.y = (-_dls_X[2]/_dls_X[0] + cen_dot.y)*scale;
    res.size.width =  res.size.height = 2 / abs(_dls_X[0]) * scale;
    res.angle = 0;


}


void ElliFit(double *data, double &error, cv::RotatedRect &res, double scale)
{
    double rp[5];
	double a1p, a2p, a11p, a22p, C2, alpha[9];
	int dot_num = data[14];
	//Mat buf_fit_data(6, 6, CV_64FC1, _buf_fit_data);
	cv::Point2d cen_dot(data[12] / (2 * dot_num), data[13] / (2 * dot_num));//the center of the fitting data.

	double A[9], B[9], D[9], C[9], B_invD_Bt[9], lambda;
	cv::Mat mB(3, 3, CV_64FC1, B), mD(3, 3, CV_64FC1, D), mB_invD_Bt(3, 3, CV_64FC1, B_invD_Bt);
	A[0] = data[0] - 2 * cen_dot.x*data[3] + 6 * cen_dot.x*cen_dot.x*data[5] - 3 * dot_num*pow(cen_dot.x, 4);
	A[3] = A[1] = data[1] - 3 * cen_dot.x*data[4] + 3 * cen_dot.x*cen_dot.x*data[8] - cen_dot.y*data[3] + cen_dot.x*cen_dot.y * 6 * data[5] - 6 * dot_num*cen_dot.y*pow(cen_dot.x, 3);
	A[6] = A[2] = data[2] - cen_dot.y*data[4] + cen_dot.y*cen_dot.y*data[5] - cen_dot.x*data[7] / 2 + cen_dot.x*cen_dot.y*data[8] * 2 + cen_dot.x*cen_dot.x*data[11] - 3 * dot_num*pow(cen_dot.x*cen_dot.y, 2);
	B[0] = data[3] - 6 * cen_dot.x*data[5] + 4 * dot_num*pow(cen_dot.x, 3);
	B[1] = data[4] - cen_dot.x*data[8] * 2 - 2 * cen_dot.y*data[5] + 4 * dot_num*cen_dot.x*cen_dot.x*cen_dot.y;
	B[2] = data[5] - dot_num*cen_dot.x*cen_dot.x;

	A[4] = 4 * A[2];
	A[7] = A[5] = data[6] - 1.5 * cen_dot.y*data[7] + 3 * cen_dot.y*cen_dot.y*data[8] - cen_dot.x*data[10] + 6 * cen_dot.x*cen_dot.y*data[11] - 6 * dot_num*cen_dot.x*pow(cen_dot.y, 3);
	B[3] = 2 * B[1];
	B[4] = data[7] - 4 * cen_dot.y*data[8] - 4 * cen_dot.x*data[11] + 8 * dot_num*cen_dot.x*cen_dot.y*cen_dot.y;
	B[5] = data[8] - 2 * dot_num*cen_dot.x*cen_dot.y;

	A[8] = data[9] - 2 * cen_dot.y*data[10] + 6 * cen_dot.y*cen_dot.y*data[11] - 3 * dot_num*pow(cen_dot.y, 4);
	B[6] = B[4] / 2;
	B[7] = data[10] - 6 * cen_dot.y*data[11] + 4 * dot_num*pow(cen_dot.y, 3);
	B[8] = data[11] - dot_num*cen_dot.y*cen_dot.y;

	D[0] = 4 * B[2];
	D[3] = D[1] = 2 * B[5];
	D[2] = 0;

	D[4] = 4 * B[8];
	D[5] = D[6] = D[7] = 0;
	D[8] = dot_num;

	mB_invD_Bt = mB*mD.inv()*mB.t();
	double *_mB_invD_Bt = (double*)mB_invD_Bt.data;
	for (int i = 0; i < 9; i++)
		C[i] = A[i] - _mB_invD_Bt[i];

	double detC, temp, detC22;
	detC = C[0] * (C[4] * C[8] - C[5] * C[7]) - C[1] * (C[3] * C[8] - C[5] * C[6]) + C[2] * (C[3] * C[7] - C[4] * C[6]);
	temp = (C[0] * C[4] - C[1] * C[1]);
	if (abs(temp) < 1e-6)
	{
		error = -1;
		return;
	}
	lambda = detC / temp;
	error = lambda;

	detC22 = C[0] * C[4] - C[3] * C[1];
	if (abs(detC22) < 1e-6)
	{
		error = -1;
		return;
	}

	alpha[0] = -(C[2] * C[4] - C[5] * C[1]) / detC22;
	alpha[1] = -(C[0] * C[5] - C[3] * C[2]) / detC22;
	alpha[2] = 1;

	Mat invD_B = mD.inv()*mB.t();
	double *_invD_B = (double*)invD_B.data;
	alpha[3] = -(_invD_B[0] * alpha[0] + _invD_B[1] * alpha[1] + _invD_B[2] * alpha[2]);
	alpha[4] = -(_invD_B[3] * alpha[0] + _invD_B[4] * alpha[1] + _invD_B[5] * alpha[2]);
	alpha[5] = -(_invD_B[6] * alpha[0] + _invD_B[7] * alpha[1] + _invD_B[8] * alpha[2]);




	double *_dls_X = alpha;
	double dls_k = _dls_X[0] * _dls_X[2] - _dls_X[1] * _dls_X[1];
	for (int i = 0; i < 6; i++)
		_dls_X[i] /= sqrt(abs(dls_k));



	rp[0] = _dls_X[1] * _dls_X[4] - _dls_X[2] * _dls_X[3];
	rp[1] = _dls_X[1] * _dls_X[3] - _dls_X[0] * _dls_X[4];
	if (fabs(_dls_X[0] - _dls_X[2]) > 1e-10)
		rp[4] = atan(2 * _dls_X[1] / (_dls_X[0] - _dls_X[2])) / 2;
	else
	{
		if (_dls_X[1] > 0)
			rp[4] = CV_PI / 4;
		else
			rp[4] = -CV_PI / 4;
	}
	//至此拟合出来的参数信息是以左上角0,0位置为原点，row为x轴，col为y轴为基准的
	a1p = cos(rp[4])*_dls_X[3] + sin(rp[4])*_dls_X[4];
	a2p = -sin(rp[4])*_dls_X[3] + cos(rp[4])*_dls_X[4];
	a11p = _dls_X[0] + tan(rp[4])*_dls_X[1];
	a22p = _dls_X[2] - tan(rp[4])*_dls_X[1];
	C2 = a1p*a1p / a11p + a2p*a2p / a22p - _dls_X[5];
	double dls_temp1 = C2 / a11p, dls_temp2 = C2 / a22p, dls_temp;
	if (dls_temp1 > 0 && dls_temp2 > 0)
	{
		rp[2] = sqrt(dls_temp1);
		rp[3] = sqrt(dls_temp2);
		if (rp[2] < rp[3])
		{
			if (rp[4] >= 0)
				rp[4] -= CV_PI / 2;
			else
				rp[4] += CV_PI / 2;
			dls_temp = rp[2];
			rp[2] = rp[3];
			rp[3] = dls_temp;
		}
	}
	else
	{
		error = -2;
		return;
	}
	res.center.x = (rp[0] + cen_dot.x)*scale;
	res.center.y = (rp[1] + cen_dot.y)*scale;
	res.size.width = 2 * rp[2] * scale;
	res.size.height = 2 * rp[3] * scale;
	res.angle = rp[4] / CV_PI * 180;
}

void fitEllipse(int _N, double*_pts, double *_shape_parms)
{
    int N = _N;
    double *p = _pts;


    const int mat_element_num = 15;
    double tmpMat[mat_element_num];
    double FitMat[mat_element_num];
    for(int i = 0; i < mat_element_num; i++)
        tmpMat[i] = FitMat[i] = 0;
    for(int i = 0; i < N; i++)
    {
        double x = p[i];
        double y = p[i + N];
        GetPointMat(x, y, PIXEL_SCALE_ELLIPSE_IMAGE, tmpMat);
        for(int k = 0; k < mat_element_num; k++)
            FitMat[k] += tmpMat[k];
    }
    double error;
    RotatedRect res;
    ElliFit(FitMat, error, res, PIXEL_SCALE_ELLIPSE_IMAGE);
    if (error < -0.5)
    {
        res.size.width = -100;
        res.size.height = -100;
    }

    double *res_elp = _shape_parms;
    res_elp[0] = res.center.x;
    res_elp[1] = res.center.y;
    res_elp[2] = res.size.width;
    res_elp[3] = res.size.height;
    res_elp[4] = res.angle;
}



void fitCircle(int _N, double*_pts, double *_shape_parms)
{
    int N = _N;
    double *p = _pts;


    const int mat_element_num = 15;
    double tmpMat[mat_element_num];
    double FitMat[mat_element_num];
    for(int i = 0; i < mat_element_num; i++)
        tmpMat[i] = FitMat[i] = 0;
    for(int i = 0; i < N; i++)
    {
        double x = p[i];
        double y = p[i + N];
        GetPointMat(x, y, PIXEL_SCALE, tmpMat);
        for(int k = 0; k < mat_element_num; k++)
            FitMat[k] += tmpMat[k];
    }
    double error;
    RotatedRect res;
    fitCircle(FitMat, error, res, PIXEL_SCALE);
    if (error < -0.5)
    {
        res.size.width = -100;
        res.size.height = -100;
    }

    double *res_elp = _shape_parms;
    res_elp[0] = res.center.x;
    res_elp[1] = res.center.y;
    res_elp[2] = res.size.width;
    res_elp[3] = res.size.height;
    res_elp[4] = res.angle;
}