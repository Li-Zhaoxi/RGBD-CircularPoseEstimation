import numpy as np
from scipy.spatial.transform import Rotation 
from ElpPy.utils import GeneralSpacialCircle, GeneralCamera, perspectCircular2Image, ElliFit, drawCirclePose
import cv2
import os
import json
from json import JSONEncoder

gt_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/data.npz'
root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender'

# gt_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneTrain/data.npz'
# root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneTrain'

data = np.load(gt_path, allow_pickle=True)
motion_postion = data['motion_postion']
motion_eulerxyz = data['motion_eulerxyz']
camK = data['camK']
gcam = GeneralCamera(camK[0, 0], camK[1, 1], camK[0, 2], camK[1, 2])


class GTEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)

# 彩色图片名
# 深度图名


num_images = len(motion_postion)
num_bits = 4
print(num_bits)

pose_data = {}
# print(num_images)
# exit()
usage_num = 0
for idx_images in range(num_images):
    # idx_images = 85
    # idx_images = 2752
    # 获取文件名
    print('idx_images = {0}'.format(idx_images))
    if idx_images < 10000:
        data_suffix = str(idx_images).zfill(num_bits)
    else:
        data_suffix = str(idx_images)
        
    
    
    color_suffix = 'rgb/COLOR_{0}.png'.format(data_suffix)
    depth_suffix = 'depth/DEPTH_{0}.exr'.format(data_suffix)
    
    each_image_data = {}
    each_image_data['color_suffix'] = color_suffix
    each_image_data['depth_suffix'] = depth_suffix
    
    # 相机内参
    gcam = GeneralCamera(camK[0, 0], camK[1, 1], camK[0, 2], camK[1, 2])
    each_image_data['camera'] = gcam.getDict()
    
    # 给出每一个潜在的空间圆的位置
    # 存储相机坐标系到世界坐标系的Trans和空间圆信息
    current_position = motion_postion[idx_images]
    current_eulerxyz = motion_eulerxyz[idx_images]
    
    # print(current_eulerxyz / np.pi * 180)
    rot = Rotation.from_euler('xyz', current_eulerxyz)
    matRot = rot.as_matrix().transpose()
    
    if matRot[2, 2] > 0:
        matRot[1, :] *= -1
        matRot[2, :] *= -1
    
    # print('matRot', matRot)
    
    # [3x3] [3x1]
    Rwc = matRot
    twc = np.matmul(-matRot, np.array([current_position]).transpose())
    
    
    usage_spacial_cirle = [] # 使用的空间圆
    usage_pixel_innersquare = [] # 在图像坐标系下，空间圆内部的方形4个点的坐标
    usage_pixel_innersquare_W = [] # 在世界坐标系下，空间圆内部的方形4个点的坐标
    usage_pixel_leftline = [] # 在图像坐标系下，空间圆内部的方形最左边直线的像素坐标
    usage_pixel_leftline_W = [] # 在世界坐标系下，空间圆内部的方形最左边直线的像素坐标
    usage_cirle_Rot = [] # 空间圆对应的完整Pose
    obj_rows = 3
    obj_cols = 5
    loc_xw = (np.array(list(range(obj_cols))) - obj_cols//2) * 4
    loc_yw = (np.array(list(range(obj_rows))) - obj_rows//2) * 4
    # print('loc_xw, loc_yw', loc_xw, loc_yw)
    
    # loc_xw = np.array([0.0, 4.0])
    # loc_yw = np.array([0.0])
    
    for each_xw in loc_xw:
        for each_yw in loc_yw:
            # 计算空间圆
            txy = np.array([[each_xw], [each_yw], [0.0]])
            each_twc = np.matmul(Rwc, txy) + twc
            gscir = GeneralSpacialCircle(Rwc[:, 2], each_twc[:, 0], 0.8)
            # print('each_twc', each_twc)
            usage_spacial_cirle.append(gscir)
            
            usage_cirle_Rot.append(Rwc)
            
            # 计算空间圆内部的矩形
            length = 0.5
            xyzw = np.array([[-length, length, 0.0],
                             [length, length, 0.0],
                             [length, -length, 0.0],
                             [-length, -length, 0.0]]).transpose()
            xyzc = np.matmul(Rwc, xyzw) + each_twc
            pts = gcam.cvtpts_cam2img(xyzc.transpose())
            if len(pts) > 0 and pts.shape[0] == 4:
                usage_pixel_innersquare.append(pts)
            else:
                usage_pixel_innersquare.append([])
            usage_pixel_innersquare_W.append(xyzw)
            
            
            # 计算每个方形最左边的直线坐标
            if len(pts) > 0 and pts.shape[0] == 4:
                usage_pixel_leftline.append([pts[0, :], pts[-1, :]])
            else:
                usage_pixel_leftline.append([])
            usage_pixel_leftline_W.append([xyzw[:, 0], xyzw[:, -1]])
            
    
    each_image_data['innersquare'] = usage_pixel_innersquare
    each_image_data['innersquare_W'] = usage_pixel_innersquare_W
    each_image_data['leftline'] = usage_pixel_leftline
    each_image_data['leftline_W'] = usage_pixel_leftline_W
    each_image_data['Rot'] = usage_cirle_Rot
    tmp_gscir = []
    for each_gscir in usage_spacial_cirle:
        tmp_gscir.append(each_gscir.getDict())
    each_image_data['spacialcircle'] = tmp_gscir
    
    
    
    color_image_path = os.path.join(root_path, color_suffix)
    if not os.path.exists(color_image_path):
        continue
    
    usage_num += 1
    
    each_image_data['size'] = np.array([1280, 720])
    
    pose_data[idx_images] = each_image_data
    
    # 绘图可视化    
    # imgC = cv2.imread(color_image_path)
    # for each_gscir, each_square_pts, each_left_line in zip(usage_spacial_cirle, usage_pixel_innersquare, usage_pixel_leftline):
    #     # print(each_gscir.getDict())
    #     drawCirclePose(imgC, gcam, each_gscir, square_pixels=np.array(each_square_pts), cooperate_line=np.array(each_left_line))
        
    # save_png_path = os.path.join(root_path, 'tmp', color_suffix)
    # print(save_png_path)
    # exit()
    # cv2.imwrite(save_png_path, imgC)
    
    # cv2.imshow('imgC', imgC)
    # cv2.waitKey(0)
    # exit()
# Blender gt to full pose gt

print('final usage num: {0} / {1}'.format(usage_num, num_images))

json_str = json.dumps(pose_data, indent=1, cls=GTEncoder)

save_path = os.path.join(root_path, 'data.json')
with open(save_path, 'w') as f:
    f.write(json_str)
    f.close()


