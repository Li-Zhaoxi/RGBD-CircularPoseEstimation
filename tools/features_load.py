import cv2
import numpy as np
import json

# 加载一个图像的Harris特征返回特征点坐标: N*2
def loadHarrisJson(json_path)->np.ndarray:
    with open(json_path, 'r') as load_f:
        corner_dict = json.load(load_f)
        load_f.close()
    
    pi = corner_dict['pi']
    pj = corner_dict['pj']
    
    pts = np.vstack([pj, pi]).transpose()
    return pts


# 加载一个图像的LSD特征返回直线段坐标: N*4
def loadLSDJson(json_path)->np.ndarray:
    with open(json_path, 'r') as load_f:
        lines_dict = json.load(load_f)
        load_f.close()
    lines = lines_dict['lines']
    return np.array(lines)
    
class ELSDData(object):
    def __init__(self) -> None:
        super().__init__()
        
        self.cx = None
        self.cy = None
        self.rx = None
        self.ry = None
        self.angle = None
        self.regs = None
        self.start = None
        self.end = None
        self.isFull = None
        self.fa = None
        self.fs = None
    
    def load_dict(self, data_dict):
        self.cx = data_dict['elp']['cx']
        self.cy = data_dict['elp']['cy']
        self.rx = data_dict['elp']['rx']
        self.ry = data_dict['elp']['ry']
        self.angle = data_dict['elp']['angle']
        
        self.regs = np.array(data_dict['regs'])
        self.start = data_dict['start']
        self.end = data_dict['end']
        self.isFull = data_dict['isFull']
        self.fa = data_dict['fa']
        self.fs = data_dict['fs']
        

def loadELSDJson(json_path):
    with open(json_path, 'r') as load_f:
        elps_dict = json.load(load_f)
        load_f.close()
        
    num_elps = elps_dict['total']
    elsd_elps = []
    for idx_elp in range(num_elps):
        elp_data = elps_dict[str(idx_elp)]
        tmp = ELSDData()
        tmp.load_dict(elp_data)
        elsd_elps.append(tmp)
    
    return elsd_elps


def loadMaskRCNN(data_path):
    data = np.load(data_path, allow_pickle=True)
    masks = data['masks']
    
    return np.array(masks)
    
def loadGTMask(data_path):
    mask = cv2.imread(data_path)
    return mask
        
        