#ifndef ELP_TOOLS_H
#define ELP_TOOLS_H

// convert ellipse shape parms to equation parms
// src: elpshape [x0,y0,a,b,theta] 
// dst: outparms [a1,a2,a3,a4,a5,a6]
void ELPShape2Equation(double *elpshape, double *outparms);
void ELPEquation2Shape(double *elpque, double *outparms);

void CalculateRangeAtY(double *elpparm, double y, double *x1, double *x2);
void CalculateRangeOfY(double *elpshape, double *x_min, double *x_max, double *y_min, double *y_max);


// Pose
void GetCirclePos(double* _C, double* _K, double Radius, double* _X1, double* _X2, double* _N1, double* _N2);


// Fitting
void fitEllipse(int _N, double*_pts, double *_shape_parms);
void fitCircle(int _N, double*_pts, double *_shape_parms);
void GeneralDirectLeastSquare(double *_dataS, double *_dataC, int dim, double *err, double *_dataX);
void MeasureCircleSampsonError(double *px, double *py, double *pz, int num, double *parms, double *error);
// Eval
void fasterCalculateOverlap(double *elp1, double *elp2, double *ration);


// colorizer
void depthColorizer(int rows, int cols, unsigned short* _data, unsigned char* _outdata);

// pre-process
void calCannyThreshold(unsigned char *_ImgG, int rows, int cols, int *low, int *high);
void findContours(unsigned char *_edge, int rows, int cols, int min_edge_num, int *contour_num, int **contours, int *each_contour_num);

#endif 