import numpy as np
cimport numpy as np


cdef extern from "MeasureTools.h":
    void ELPShape2Equation(double *, double *)
    void CalculateRangeAtY(double *, double, double *, double *)
    void CalculateRangeOfY(double *, double *, double *, double *, double *)
    void GetCirclePos(double*, double*, double, double*, double*, double*, double*)
    void fitCircle(int N, double*_pts, double *_shape_parms)
    void fitEllipse(int N, double*_pts, double *_shape_parms)

def pyELPShape2Equation(np.ndarray[np.float64_t, ndim=1] shapeparms):
    cdef np.ndarray[np.float64_t, ndim=1] equparms = np.zeros(shape=(6,), dtype=np.float64)
    ELPShape2Equation(&shapeparms[0], &equparms[0])
    return equparms

def pyCalculateRangeOfY(np.ndarray[np.float64_t, ndim=1] shapeparms):
    cdef double xmin = np.nan
    cdef double xmax = np.nan
    cdef double ymin = np.nan
    cdef double ymax = np.nan
    CalculateRangeOfY(&shapeparms[0], &xmin, &xmax, &ymin, &ymax)
    return xmin, xmax, ymin, ymax

def pyCalculateRangeAtY(np.ndarray[np.float64_t, ndim=1] shapeparms, double y):
    cdef double xmin = np.nan
    cdef double xmax = np.nan
    CalculateRangeAtY(&shapeparms[0], y, &xmin, &xmax)
    if xmin > xmax:
        return None
    else:
        return xmin, xmax

def pyGetCirclePos(np.ndarray[np.float64_t, ndim=2] C, np.ndarray[np.float64_t, ndim=2] K, double Radius):
    assert C.shape[0] == 3 and C.shape[1] == 3 and K.shape[0] == 3 and K.shape[1] == 3

    cdef np.ndarray[np.float64_t, ndim=1] X1 = np.zeros(shape=(3,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] X2 = np.zeros(shape=(3,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] N1 = np.zeros(shape=(3,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] N2 = np.zeros(shape=(3,), dtype=np.float64)

    GetCirclePos(&C[0,0], &K[0,0], Radius, &X1[0], &X2[0], &N1[0], &N2[0])

    return X1, X2, N1, N2

def pyFitCircle(np.ndarray[np.float64_t, ndim=2] pts):
    cdef np.ndarray[np.float64_t, ndim=1] fitres = np.zeros(shape=(5,), dtype=np.float64)
    cdef int N = pts.shape[1]
    cdef int dim = pts.shape[0]
    #assert N > dim and N > 4
    fitCircle(N, &pts[0,0], &fitres[0])
    return fitres

def pyFitEllipse(np.ndarray[np.float64_t, ndim=2] pts):
    cdef np.ndarray[np.float64_t, ndim=1] fitres = np.zeros(shape=(5,), dtype=np.float64)
    cdef int N = pts.shape[1]
    cdef int dim = pts.shape[0]
    #assert N > dim and N > 4
    fitEllipse(N, &pts[0,0], &fitres[0])
    return fitres