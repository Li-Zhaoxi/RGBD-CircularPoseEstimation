from ElpPy.dataproc import GTPoseLoader
import os
import cv2 
import numpy as np
from PIL import Image
import json
from ElpPy.utils import GeneralEllipse, GeneralLine, GeneralLineSegment, drawCirclePose, perspectCircular2Image
from tools.features_load import loadELSDJson, loadGTMask, loadHarrisJson, loadLSDJson, loadMaskRCNN
from tqdm import tqdm
from ElpPy.utils import findMatchEllipse, findMatchMask, findMatchLineSegment, findMatchSquareCorners, drawPerspectiveCircle

# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingLight3/'
dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark3/'
dataset_name = 'CircularPose-RealRing40Traj'

loader = GTPoseLoader()
loader.load(dataset_root_path, dataset_name)

gt_num = len(loader)
print(dataset_root_path)
print(gt_num)

# ELSD算法存在1个像素的偏移
ellipse_pixel_offset = 1

# 记录每一张图
#  记录存在匹配的结果，分为features和gt
# all_match_data = {}
for idx_image in tqdm(range(0, gt_num)):
    # idx_image = 3
    # print('Processing {0}th image.'.format(idx_image))
    # idx_image = 10
    gt = loader.__getitem__(idx_image, False)
    
    color_full_path = gt['color_full_path']
    root_path = gt['root_path']
    
    # print(color_full_path)
    
    color_name = os.path.basename(color_full_path)
    
    usage_match_data = {}
    
    # 加载真值信息
    usage_gelp = gt['gtellipse']
    # print(usage_gelp.ellipse_shape_img())
    usage_innersquare = gt['innersquare']
    usage_innersquare_W = gt['innersquare_W']
    usage_leftlines = gt['leftline'] # K * 2 * 2
    usage_leftlines_W = gt['leftline_W'] # K * 2 * 2
    usage_gcam = gt['camera']
    end_pose = gt['end_pose']
    joint_angle = gt['joint_angle']
    
        
    # 加载检测信息
    ELSD_Json_path = os.path.join(root_path, 'ExtractedFeatures/ELSD', color_name + '.json')
    det_elps = loadELSDJson(ELSD_Json_path)
    # print(det_elps)
    
    mask_data_path = os.path.join(root_path, 'ExtractedFeatures/MaskRCNN', color_name + '.npz')
    det_masks = loadMaskRCNN(mask_data_path)
    mask_gt_path = gt['mask_full_path']
    gt_masks = loadGTMask(mask_gt_path)
    
    LSD_Json_path = os.path.join(root_path, 'ExtractedFeatures/LSD', color_name + '.json')
    det_lsd = loadLSDJson(LSD_Json_path)
    
    Harris_Json_path = os.path.join(root_path, 'ExtractedFeatures/Harris', color_name + '.json')
    det_corners = loadHarrisJson(Harris_Json_path)
    
    
    ELSDthin_path = os.path.join(root_path, 'ExtractedFeatures/ELSDthin', str(idx_image) + '.png')
    
    
    # 设置匹配阈值
    T_elp = 10
    T_corner = 8
    T_line = 8
    
    imgC = cv2.imread(color_full_path)
    
    
    each_image_match_data = {} 
    # 真值就1个
    
    
    # 匹配椭圆
    found_ellipse = None
    found_ellipticl_pts = None
    # print(len(det_elps))
    match_res = findMatchEllipse(usage_gelp, det_elps, T_elp, ellipse_pixel_offset)
    if match_res is not None:
        found_ellipse = match_res[0]
        found_ellipticl_pts = match_res[1]
    
    elsdthin = cv2.imread(ELSDthin_path)
    found_ellipticl_thinpts = np.argwhere(elsdthin > 100)
    
    # print(found_ellipse,usage_gelp)
    # exit()
    
    # 匹配Mask
    found_mask = findMatchMask(gt_masks, det_masks, 0)
    
    # 匹配直线段
    found_linesements = findMatchLineSegment(usage_leftlines, det_lsd, T_line)
    
    # 匹配角点
    found_corners = findMatchSquareCorners(usage_innersquare, det_corners, T_corner)
    
    
    
    each_gt_match_data = {}
    each_gt_match_data['elliptical_pixels'] = found_ellipticl_pts
    each_gt_match_data['elliptical_thinpixels'] = found_ellipticl_thinpts[:, [1, 0]]
    each_gt_match_data['ellipse'] = found_ellipse
    each_gt_match_data['gtellipse'] = usage_gelp
    each_gt_match_data['mask_pts'] = found_mask
    each_gt_match_data['square_pixels'] = found_corners
    each_gt_match_data['square_pixels_W'] = usage_innersquare_W
    each_gt_match_data['cooperate_line'] = found_linesements
    each_gt_match_data['cooperate_line_W'] = usage_leftlines_W
    each_gt_match_data['end_pose'] = end_pose
    each_gt_match_data['joint_angle'] = joint_angle
    
    each_image_match_data[0] = each_gt_match_data
    each_image_match_data['camera'] = usage_gcam.getDict()
    
    # exit()
    
    
    # # 可视化部分    
    # drawPerspectiveCircle(imgC, found_ellipse, found_ellipticl_pts, found_corners, found_linesements, None, found_mask)
    # imgT = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
    # show_image = Image.fromarray(imgT)
    # show_image.show()
    # exit()
    
    save_npz_path = os.path.join(dataset_root_path, 'FeaturesMatches', '{0}'.format(idx_image))
    np.savez(save_npz_path, data=each_image_match_data)
    
    