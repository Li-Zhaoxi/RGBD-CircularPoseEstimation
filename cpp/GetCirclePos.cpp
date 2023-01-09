#include "MeasureTools.h"

void GetCirclePos(cv::Mat C, cv::Mat K, double Radius, cv::Mat &X1, cv::Mat &X2, cv::Mat &N1, cv::Mat &N2)
{
    cv::Mat C2;
	C2 = K.t() * C * K;

	double* _dC2 = (double*)C2.data;
	double min_val = 1e4;
	for (int i = 0; i < 9; i++)
	{
		if (std::abs(_dC2[i]) > 1e-4)
		{
			if (_dC2[i] < min_val)
				min_val = _dC2[i];
		}
	}
	C2 = C2 / min_val;

	cv::Mat V, D;
	cv::eigen(C2, D, V);
	V = V.t();
	//std::cout << "D" << D << std::endl;
	//std::cout << "V" << V << std::endl;
	double d1 = D.at<double>(0), d2 = D.at<double>(1), d3 = D.at<double>(2);
	double lamda1, lamda2, lamda3, change(0), change2(0);

	cv::Mat e1, e2, e3;

	if (d1*d2 > 0)
	{
		if (abs(d1) > abs(d2))
		{
			lamda1 = d1;
			lamda2 = d2;
			e1 = V.col(0);
			e2 = V.col(1);
			change = 0;
		}
		else
		{
			lamda1 = d2;
			lamda2 = d1;
			e1 = V.col(1);
			e2 = V.col(0);
			change = 1;
		}
		lamda3 = d3;
		e3 = V.col(2);
	}
	else if (d1*d3 > 0)
	{
		if (abs(d1) > abs(d3))
		{
			lamda1 = d1;
			lamda2 = d3;
			e1 = V.col(0);
			e2 = V.col(2);
			change = 1;
		}
		else
		{
			lamda1 = d3;
			lamda2 = d1;
			e1 = V.col(2);
			e2 = V.col(0);
			change = 2;
		}
		lamda3 = d2;
		e3 = V.col(1);
	}
	else
	{
		if (abs(d2) > abs(d3))
		{
			lamda1 = d2;
			lamda2 = d3;
			e1 = V.col(1);
			e2 = V.col(2);
			change = 2;
		}
		else
		{
			lamda1 = d3;
			lamda2 = d2;
			e1 = V.col(2);
			e2 = V.col(1);
			change = 1;
		}
		lamda3 = d1;
		e3 = V.col(0);
	}

	if (e3.at<double>(2) < 0)
	{
		e3 = -e3;
		change2 = 1;
	}

	e1 = e2.cross(e3);
	cv::Mat P(3, 3, CV_64FC1);
	e1.copyTo(P.col(0)), e2.copyTo(P.col(1)), e3.copyTo(P.col(2));
	//std::cout << "e1" << e1 << std::endl;
	//std::cout << "e2" << e2 << std::endl;
	//std::cout << "e3" << e3 << std::endl;
	//std::cout << "P" << P << std::endl;

	lamda1 = abs(lamda1), lamda2 = abs(lamda2), lamda3 = abs(lamda3);

	cv::Mat xo1(3, 1, CV_64FC1), n1(3, 1, CV_64FC1), xo2(3, 1, CV_64FC1), n2(3, 1, CV_64FC1);
	xo1.at<double>(0) = Radius * sqrt((lamda3 / lamda1) * ((lamda1 - lamda2) / (lamda1 + lamda3)));
	xo1.at<double>(1) = 0;
	xo1.at<double>(2) = Radius * sqrt((lamda1 / lamda3) * ((lamda2 + lamda3) / (lamda1 + lamda3)));
	n1.at<double>(0) = sqrt(((lamda1 - lamda2) / (lamda1 + lamda3)));
	n1.at<double>(1) = 0;
	n1.at<double>(2) = -sqrt(((lamda2 + lamda3) / (lamda1 + lamda3)));
	xo2.at<double>(0) = -xo1.at<double>(0);
	xo2.at<double>(1) = xo1.at<double>(1);
	xo2.at<double>(2) = xo1.at<double>(2);
	n2.at<double>(0) = -n1.at<double>(0);
	n2.at<double>(1) = n1.at<double>(1);
	n2.at<double>(2) = n1.at<double>(2);


	X1 = P * xo1;
	X2 = P * xo2;
	N1 = P * n1;
	N2 = P * n2;
}


void GetCirclePos(double* _C, double* _K, double Radius, double* _X1, double* _X2, double* _N1, double* _N2)
{
    cv::Mat X1, X2, N1, N2;
    cv::Mat C(3,3,CV_64FC1, _C), K(3,3,CV_64FC1, _K);
    GetCirclePos(C, K, Radius, X1, X2, N1, N2);
    memcpy(_X1, X1.data, sizeof(double) * 3);
    memcpy(_X2, X2.data, sizeof(double) * 3);
    memcpy(_N1, N1.data, sizeof(double) * 3);
    memcpy(_N2, N2.data, sizeof(double) * 3);
}