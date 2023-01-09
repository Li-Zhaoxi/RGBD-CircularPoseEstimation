from ElpPy.dataproc import GTPoseLoader
import os
import cv2 
import numpy as np
from PIL import Image
import json
from ElpPy.utils import GeneralCamera, GeneralEllipse, GeneralLine, GeneralLineSegment, GeneralSpacialCircle, drawCirclePose, perspectCircular2Image
from tools.features_load import loadELSDJson, loadGTMask, loadHarrisJson, loadLSDJson, loadMaskRCNN

dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
dataset_name = 'CircularPose-15PlaneBlender'
loader = GTPoseLoader()
loader.load(dataset_root_path, dataset_name)

gt_num = len(loader)
print(gt_num)

for idx_image in range(gt_num):
    idx_image = 144
    print('Processing {0}th image.'.format(idx_image))
    gt = loader.__getitem__(idx_image, False)
    
    color_full_path = gt['color_full_path']
    matches_full_path = os.path.join(dataset_root_path, 'FeaturesMatches', '{0}.npz'.format(idx_image))
    
    each_image_match_data = np.load(matches_full_path, allow_pickle=True)['data'][()]

    # print(each_image_match_data[()])
    # each_image_match_data = each_image_match_data['data']
    # print(each_image_match_data)
    # print(each_image_match_data[()][7])
    # print(list(each_image_match_data[()].keys()))
    
    # print(type(each_image_match_data))
    
    # exit()
    camera_dict = each_image_match_data['camera']
    gcam = GeneralCamera.loadDict(camera_dict)
    
    usage_keys = list(each_image_match_data.keys())
    
    usage_gscir = gt['spacialcircle']
    total_gt = len(usage_gscir) # 理论真值个数
    imgC = cv2.imread(color_full_path)
    
    
    for idx_gt in range(total_gt):
        if idx_gt not in usage_keys:
            continue
        
        each_gt_match_data = each_image_match_data[idx_gt]
        
        gtcircle = GeneralSpacialCircle(1, 1, 1)
        gtcircle.loadDict(each_gt_match_data['gtcircle'])
        elliptical_pixels = each_gt_match_data['elliptical_pixels'] # N * 2
        
        gelp = each_gt_match_data['ellipse']
        # print(each_gt_match_data['ellipse'])
        # gelp.loadData(each_gt_match_data['ellipse'])
        
        mask_pts = each_gt_match_data['mask_pts'] # N * 2
        
        square_pixels = each_gt_match_data['square_pixels'] # 4 * 2
        
        cooperate_line = each_gt_match_data['cooperate_line'] # 2 * 2
        
        drawCirclePose(imgC, gcam, gtcircle, 
                       elliptical_pixels=elliptical_pixels, square_pixels=square_pixels, other_pts=None,
                       mask_pts=mask_pts, cooperate_line=cooperate_line, ingelp=gelp)
        
    # cv2.imshow('imgC', imgC)
    # cv2.waitKey()
    
    imgT = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
    show_image = Image.fromarray(imgT)
    show_image.show()
    exit()
        
        
    
    