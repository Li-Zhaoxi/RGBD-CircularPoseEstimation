import csv
import numpy as np
from ElpPy.plane_fitting.plane_proc import fit_plane
from ElpPy.utils import get_norm_Rotation
import circle_fit as cirf



def load_csv_data(csv_path):
    data =[]
    with open(csv_path)as f:
        f_csv = csv.reader(f)
        row_index = -1
        for row in f_csv:
            row_index += 1
            if row_index == 0 or row_index == 1:
                continue
            
            tmp = [float(each_col) for each_col in row[36:44]]
            data.append(tmp)
    
    data = np.array(data)
    
    return data

csv_path_plane = '/home/expansion/lizhaoxi/codes-lzx/perspective-circle-depth/NDICalibration/data/plane_1.csv'
csv_path_circle = '/home/expansion/lizhaoxi/codes-lzx/perspective-circle-depth/NDICalibration/data/circle_1.csv'

data_plane = load_csv_data(csv_path_plane)
print(data_plane)
data_circle = load_csv_data(csv_path_circle)


## 平面拟合，误差分析
init_circle_norm, d, err, pts_center = fit_plane(data_plane[:, 4:7])
print('init_circle_norm', init_circle_norm)
print('d', d)
print('err', err, np.mean(err))
print('pts_center', pts_center)

## 空间圆拟合，误差分析
circular_pts = data_circle[:, 4:7]
circle_mean_pts = np.mean(circular_pts, axis=0)
Rot = get_norm_Rotation(init_circle_norm)
k = (d - np.dot(init_circle_norm, circle_mean_pts)) / np.linalg.norm(init_circle_norm)
pt_plane = k * init_circle_norm + circle_mean_pts # 在平面上的一个点

cirptsOnPlane = np.matmul(Rot.transpose(), (circular_pts - pt_plane).transpose())
px = cirptsOnPlane[0, :]
py = cirptsOnPlane[1, :]
xycp = np.array([px, py])
xo, yo, cirR, var = cirf.hyper_fit(xycp.transpose())
circle_fit_errors = np.abs(np.sqrt((px - xo)**2 + (py - yo)**2) - cirR)

print(circle_fit_errors)
print('center', np.array([xo, yo]))
print('radius', cirR)
print('errs mean: {0}, min: {1}, max: {2}'.format(np.mean(circle_fit_errors), np.min(circle_fit_errors), np.max(circle_fit_errors)))


circle_center_c = np.matmul(Rot, np.array([[xo], [yo], [0]])).transpose()[0] + pt_plane
print('spacial circle center: ', circle_center_c)
# 空间点转到平面坐标系上

# R = get_norm_Rotation(init_circle_norm)

# px, py, R, t = self.perspective_pixels2plane(p_eI, init_circle_norm, d)
# cir_parms = self.fit_circle(px, py)
# cir_pos = np.array([[cir_parms[0], cir_parms[1], 0]]).transpose()
# init_circle_pos = (np.matmul(R, cir_pos).transpose() + t)[0]
# init_R = cir_parms[2]

# row = f_csv

# print(f_csv)