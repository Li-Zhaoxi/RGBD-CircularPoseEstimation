#include "ElpTools.h"

#include <iostream>
#include <cmath>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;

void ELPShape2Equation(double *elpshape, double *outparms)
{
	double xc, yc, a, b, theta;
	xc = elpshape[0], yc = elpshape[1], a = elpshape[2], b = elpshape[3], theta = elpshape[4];

	double parm[6];
    

	parm[0] = cos(theta)*cos(theta) / (a*a) + pow(sin(theta), 2) / (b*b);
	parm[1] = -(sin(2 * theta)*(a*a - b*b)) / (2 * a*a*b*b);
	parm[2] = pow(cos(theta), 2) / (b*b) + pow(sin(theta), 2) / (a*a);
	parm[3] = (-a*a*xc*pow(sin(theta), 2) + a*a*yc*sin(2 * theta) / 2) / (a*a*b*b) - (xc*pow(cos(theta), 2) + yc*sin(2 * theta) / 2) / (a*a);
	parm[4] = (-a*a*yc*pow(cos(theta), 2) + a*a*xc*sin(2 * theta) / 2) / (a*a*b*b) - (yc*pow(sin(theta), 2) + xc*sin(2 * theta) / 2) / (a*a);
	parm[5] = pow(xc*cos(theta) + yc*sin(theta), 2) / (a*a) + pow(yc*cos(theta) - xc*sin(theta), 2) / (b*b) - 1;

	double k = parm[0] * parm[2] - parm[1] * parm[1];

	for (int i = 0; i < 6; i++)
		outparms[i] = parm[i] / sqrt(abs(k));

}

void ELPEquation2Shape(double *elpque, double *outparms)
{
    double _elpque[6], rp[5];
    double a1p, a2p, a11p, a22p, C2, alpha[9];
    double dls_k = elpque[0] * elpque[2] - elpque[1] * elpque[1];
    for (int i = 0; i < 6; i++)
		_elpque[i] = elpque[i] / sqrt(abs(dls_k));

    rp[0] = _elpque[1] * _elpque[4] - _elpque[2] * _elpque[3];
	rp[1] = _elpque[1] * _elpque[3] - _elpque[0] * _elpque[4];
	if (fabs(_elpque[0] - _elpque[2]) > 1e-10)
		rp[4] = atan(2 * _elpque[1] / (_elpque[0] - _elpque[2])) / 2;
	else
	{
		if (_elpque[1] > 0)
			rp[4] = CV_PI / 4;
		else
			rp[4] = -CV_PI / 4;
	}
	//至此拟合出来的参数信息是以左上角0,0位置为原点，row为x轴，col为y轴为基准的
	a1p = cos(rp[4])*_elpque[3] + sin(rp[4])*_elpque[4];
	a2p = -sin(rp[4])*_elpque[3] + cos(rp[4])*_elpque[4];
	a11p = _elpque[0] + tan(rp[4])*_elpque[1];
	a22p = _elpque[2] - tan(rp[4])*_elpque[1];
	C2 = a1p*a1p / a11p + a2p*a2p / a22p - _elpque[5];
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
        for(int i = 0; i < 5; i++)
            outparms[i] = -1;
		return;
	}
    for(int i = 0; i < 5; i++)
        outparms[i] = rp[i];
}


void CalculateRangeAtY(double *elpparm, double y, double *x1, double *x2)
{
	double A, B, C, D, E, F;
	A = elpparm[0], B = elpparm[1], C = elpparm[2];
	D = elpparm[3], E = elpparm[4], F = elpparm[5];

	double Delta = pow(B*y + D, 2) - A*(C*y*y + 2 * E*y + F);

	if (Delta < 0)
		*x1 = -10, *x2 = -20;
	else
	{
        double t1, t2;
		t1 = (-(B*y + D) - sqrt(Delta)) / A;
		t2 = (-(B*y + D) + sqrt(Delta)) / A;
		
		if (t2 < t1)
			swap(t1, t2);
        
        *x1 = t1;
        *x2 = t2;
	}
}

void CalculateRangeOfY(double *elpshape, double *x_min, double *x_max, double *y_min, double *y_max)
{
    double elp_equ[6];
    ELPShape2Equation(elpshape, elp_equ);
    
    double B, C;
    B = elp_equ[1] * elp_equ[3] - elp_equ[0] * elp_equ[4];
    C = elp_equ[3] * elp_equ[3] - elp_equ[0] * elp_equ[5];
    
    double tx_min, tx_max, ty_min, ty_max;
    
    ty_min = B - sqrt(B*B + C);
    ty_max = B + sqrt(B*B + C);
    
    tx_min = -(elp_equ[1] * ty_min + elp_equ[3]) / elp_equ[0];
    tx_max = -(elp_equ[1] * ty_max + elp_equ[3]) / elp_equ[0];
    
    *x_min = tx_min;
    *x_max = tx_max;
    *y_min = ty_min;
    *y_max = ty_max;
}