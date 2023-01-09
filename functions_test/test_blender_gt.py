import numpy as np
from scipy.spatial.transform import Rotation 
from ElpPy.utils import GeneralSpacialCircle, GeneralCamera, perspectCircular2Image, ElliFit
import cv2
# rot = Rotation.from_euler('xyz', np.array([0.0, 30.0, 60.0]) / 180.0 * np.pi)

# matRot = rot.as_matrix()

# print(matRot)


def drawCirclePose(imgC: np.ndarray, gcam: GeneralCamera, gscir: GeneralSpacialCircle):
    
    # 生成圆的采样点
    xyzc = gscir.generateCircularSamples()
    
    pts = gcam.cvtpts_cam2img(xyzc)
    
    # for each_pt in pts:
    #     each_pt = np.round(each_pt).astype(int)
    #     cv2.circle(imgC, (each_pt[0], each_pt[1]), 2, (0, 0, 255))
        
    gelp = perspectCircular2Image(gcam, gscir)
    
    print(gelp.all_elp_type)
    
    # tgelp = ElliFit(pts, 'shape_image')
    
    # print(tgelp.all_elp_type)
    
    gelp.drawEllipse(imgC, (255, 0, 0), 2)



depth_path = '/home/expansion/lizhaoxi/datasets/Pose/Circular Pose - 15Plane Blender/depth/DEPTH_0114.exr'
color_path = '/home/expansion/lizhaoxi/datasets/Pose/Circular Pose - 15Plane Blender/rgb/COLOR_0114.png'
idx_frame = 114


gt_path = '/home/expansion/lizhaoxi/datasets/Pose/Circular Pose - 15Plane Blender/data.npz'

data = np.load(gt_path, allow_pickle=True)
motion_postion = data['motion_postion']
motion_eulerxyz = data['motion_eulerxyz']
camK = data['camK']



current_position = motion_postion[idx_frame]
current_eulerxyz = motion_eulerxyz[idx_frame]



rot = Rotation.from_euler('xyz', current_eulerxyz)

matRot = rot.as_matrix()
if matRot[2, 2] > 0:
    matRot[:, 1] *= -1
    matRot[:, 2] *= -1
    # matRot[:, 1] *= -1
    # matRot[:, 2] *= -1

print(matRot)


loc = np.matmul(-matRot.transpose(), np.array([current_position]).transpose()).transpose()[0]
print('loc', loc, 'current_position', current_position)
# exit()
imgC = cv2.imread(color_path)
gcam = GeneralCamera(camK[0, 0], camK[1, 1], camK[0, 2], camK[1, 2])
gscir = GeneralSpacialCircle(matRot[2, :], loc, 0.8)
drawCirclePose(imgC, gcam, gscir)

cv2.imshow('imgC', imgC)
cv2.waitKey()

