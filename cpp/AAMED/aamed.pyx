import numpy as np
cimport numpy as np
import cv2

cdef extern from "FLED.h":
    cdef cppclass FLED:
        FLED(int, int)
        int run_FLED(unsigned char*, int, int)
        void release()
        void drawAAMED(unsigned char*, int, int)
        void SetParameters(double, double, double)
        void UpdateResults(float*)
        void getEllipseEdgePointsPy(int, unsigned int*)
        int getEllipseEdgePointsNum(int)


cdef class pyAAMED:
    cdef FLED* _fled;
    cdef int drows;
    cdef int dcols;
    def __cinit__(self, int drows, int dcols):
        self._fled = new FLED(drows, dcols)
        self.drows = drows
        self.dcols = dcols

    def run_AAMED(self, np.ndarray[np.uint8_t, ndim=2] imgG):
        cdef int rows = imgG.shape[0]
        cdef int cols = imgG.shape[1]
        assert rows < self.drows and cols < self.dcols, \
            'The size ({:d}, {:d}) of an input image must be smaller than ({:d}, {:d})'.format(rows, cols, self.drows, self.dcols)
        cdef int det_num = 0
        det_num = self._fled.run_FLED(&imgG[0, 0], rows, cols)
        if det_num == 0:
            return []
        cdef np.ndarray[np.float32_t, ndim=2] detEllipse = np.zeros(shape=(det_num, 6), dtype=np.float32)
        self._fled.UpdateResults(&detEllipse[0, 0])
        return detEllipse




    def release(self):
        #self._fled.release()
        del self._fled

    def drawAAMED(self, np.ndarray[np.uint8_t, ndim=2] imgG):
        cdef int rows = imgG.shape[0]
        cdef int cols = imgG.shape[1]
        self._fled.drawAAMED(&imgG[0, 0], rows, cols)

    def setParameters(self, double theta_fsa, double length_fsa, double T_val):
        self._fled.SetParameters(theta_fsa, length_fsa, T_val)

    def getEdgePoints(self, int idx):
        cdef int pts_num = self._fled.getEllipseEdgePointsNum(idx)
        cdef np.ndarray[np.uint32_t, ndim=2] pts = np.zeros(shape=(pts_num, 2), dtype=np.uint32)
        cdef unsigned int* _pts= <unsigned int*>&pts[0, 0]
        self._fled.getEllipseEdgePointsPy(idx,_pts)
        return pts