import numpy as np
cimport numpy as np



cdef extern from "cpps/ElpTools.h":
    void ELPShape2Equation(double *, double *)
    void ELPEquation2Shape(double *, double *)
    void CalculateRangeAtY(double *, double, double *, double *)
    void CalculateRangeOfY(double *, double *, double *, double *, double *)
    void GetCirclePos(double*, double*, double, double*, double*, double*, double*)
    void fitEllipse(int N, double*_pts, double *_shape_parms)
    void fitCircle(int _N, double*_pts, double *_shape_parms)
    void MeasureCircleSampsonError(double *px, double *py, double *pz, int num, double *parms, double *error)
    void GeneralDirectLeastSquare(double *_dataS, double *_dataC, int dim, double *err, double *_dataX)
    void fasterCalculateOverlap(double *elp1, double *elp2, double *ration)
    void depthColorizer(int rows, int cols, unsigned short* _data, unsigned char* _outdata)
    void calCannyThreshold(unsigned char *_ImgG, int rows, int cols, int *low, int *high)
    void findContours(unsigned char *_edge, int rows, int cols, int min_edge_num, int *contour_num, int **contours, int *each_contour_num)



def pyELPShape2Equation(np.ndarray[np.float64_t, ndim=1] shapeparms):
    cdef int N = shapeparms.shape[0]
    assert N == 5
    cdef np.ndarray[np.float64_t, ndim=1] equparms = np.zeros(shape=(6,), dtype=np.float64)
    ELPShape2Equation(&shapeparms[0], &equparms[0])
    return equparms

def pyELPEquation2Shape(np.ndarray[np.float64_t, ndim=1] equparms):
    cdef int N = equparms.shape[0]
    assert N == 6
    cdef np.ndarray[np.float64_t, ndim=1] shapeparms = np.zeros(shape=(5,), dtype=np.float64)
    ELPEquation2Shape(&equparms[0], &shapeparms[0])
    return shapeparms

def pyCalculateRangeOfY(np.ndarray[np.float64_t, ndim=1] shapeparms):
    cdef double xmin = np.nan
    cdef double xmax = np.nan
    cdef double ymin = np.nan
    cdef double ymax = np.nan

    cdef int N = shapeparms.shape[0]
    assert N == 5
    CalculateRangeOfY(&shapeparms[0], &xmin, &xmax, &ymin, &ymax)
    return xmin, xmax, ymin, ymax

def pyCalculateRangeAtY(np.ndarray[np.float64_t, ndim=1] equparms, double y):
    cdef double xmin = np.nan
    cdef double xmax = np.nan

    cdef int N = equparms.shape[0]
    assert N == 6
    CalculateRangeAtY(&equparms[0], y, &xmin, &xmax)
    if xmin > xmax:
        return None
    else:
        return xmin, xmax


def pyGetCirclePos(np.ndarray[np.float64_t, ndim=2] C, np.ndarray[np.float64_t, ndim=2] K, double Radius):
    assert C.shape[0] == 3 and C.shape[1] == 3 and K.shape[0] == 3 and K.shape[1] == 3
    assert Radius > 0

    cdef np.ndarray[np.float64_t, ndim=1] X1 = np.zeros(shape=(3,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] X2 = np.zeros(shape=(3,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] N1 = np.zeros(shape=(3,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] N2 = np.zeros(shape=(3,), dtype=np.float64)

    GetCirclePos(&C[0,0], &K[0,0], Radius, &X1[0], &X2[0], &N1[0], &N2[0])

    return X1, X2, N1, N2


def pyFitEllipse(np.ndarray[np.float64_t, ndim=2] pts):
    cdef np.ndarray[np.float64_t, ndim=1] fitres = np.zeros(shape=(5,), dtype=np.float64)
    cdef int N = pts.shape[0]
    cdef int dim = pts.shape[1]
    assert N > 5 and dim == 2
    fitEllipse(N, &pts[0,0], &fitres[0])
    return fitres

def pyFitCircle(np.ndarray[np.float64_t, ndim=2] pts):
    cdef np.ndarray[np.float64_t, ndim=1] fitres = np.zeros(shape=(5,), dtype=np.float64)
    cdef int N = pts.shape[0]
    cdef int dim = pts.shape[1]
    assert N > 5 and dim == 2
    fitCircle(N, &pts[0,0], &fitres[0])
    return fitres

def pyGeneralDirectLeastSquare(np.ndarray[np.float64_t, ndim=2] S, np.ndarray[np.float64_t, ndim=2] C):
    cdef int Srows = S.shape[0]
    cdef int Scols = S.shape[1] 
    cdef int Crows = C.shape[0] 
    cdef int Ccols = C.shape[1]  
    assert Srows == Scols and Crows == Ccols and Srows == Crows

    cdef int dim = Srows
    cdef double err = -3.0
    cdef np.ndarray[np.float64_t, ndim=1] dataX = np.zeros(shape=(dim,), dtype=np.float64)

    GeneralDirectLeastSquare(&S[0, 0], &C[0, 0], dim, &err, &dataX[0])

    return dataX, err

# void MeasureSampsonError(double *px, double *py, double *pz, int num, double *parms, double *error)
def pyMeasureCircleSampsonError(np.ndarray[np.float64_t, ndim=1] parms,
                        np.ndarray[np.float64_t, ndim=1] px, 
                        np.ndarray[np.float64_t, ndim=1] py, 
                        np.ndarray[np.float64_t, ndim=1] pz):
    assert parms.shape[0] == 4
    cdef int pt_num = px.shape[0]
    assert px.shape[0] == py.shape[0] and px.shape[0] == pz.shape[0]
    
    cdef double error = 0
    MeasureCircleSampsonError(&px[0], &py[0], &pz[0], pt_num, &parms[0], &error)

    return error

def pyfasterCalculateOverlap(np.ndarray[np.float64_t, ndim=1] elp1, np.ndarray[np.float64_t, ndim=1] elp2):
    cdef int d1 = elp1.shape[0]
    cdef int d2 = elp2.shape[0]
    assert d1 == 5 and d2 == 5

    cdef double ration = -1

    fasterCalculateOverlap(&elp1[0], &elp2[0], &ration)

    return ration

def pyDepthColorizer(np.ndarray[np.uint16_t, ndim=2] imgD):
    cdef int rows = imgD.shape[0]
    cdef int cols = imgD.shape[1]
    assert rows > 0 and cols > 0

    cdef np.ndarray[np.uint8_t, ndim=3] res = np.zeros(shape=(rows, cols, 3), dtype='uint8')
    depthColorizer(rows, cols, &imgD[0, 0], &res[0, 0, 0])
    return res


def pycalCannyThreshold(np.ndarray[np.uint8_t, ndim=2] imgG):
    cdef int rows = imgG.shape[0]
    cdef int cols = imgG.shape[1]
    assert rows > 0 and cols > 0

    cdef int high = 10
    cdef int low = 0

    calCannyThreshold(&imgG[0, 0], rows, cols, &low, &high)

    return low, high


'''
def pyfindContours(np.ndarray[np.uint8_t, ndim=2] imgedge, int min_edge_num):
    cdef int rows = imgedge.shape[0]
    cdef int cols = imgedge.shape[1]
    assert rows > 0 and cols > 0

    cdef int total_contour_num = 0
    cdef int* each_contour_num = NULL
    cdef int** all_contours = NULL

   findContours(&imgedge[0, 0], rows, cols, min_edge_num, &total_contour_num, all_contours, each_contour_num)

    cdef list ext_contours = []
    if total_contour_num > 0:
        ext_contours = [] * total_contour_num
        for idx_contour in range(total_contour_num):
            cdef np.ndarray[np.float64_t, ndim=1] tmp = np.zeros(shape=(5,), dtype=np.float64)
            tmp = [all_contours[idx_contour][2 * idx_pt], all_contours[idx_contour][2 * idx_pt + 1] for idx_pt in range(each_contour_num[idx_contour])]
            ext_contours.append(tmp)
    
    return ext_contours
'''
 


