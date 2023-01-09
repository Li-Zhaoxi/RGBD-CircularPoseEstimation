import numpy as np
from PyQt5.QtCore import QPointF

def cvtpts2Qptfs(pts:np.ndarray):
    qpts = []
    for each_pt in pts:
        qpts.append(QPointF(each_pt[0], each_pt[1]))
        
    return qpts

def cvtQPtfs2pts(qpts)->np.ndarray:
    pts = []
    for each_qpt in qpts:
        pts.append([each_qpt.x(), each_qpt.y()])
        
    return np.array(pts, dtype=np.float)