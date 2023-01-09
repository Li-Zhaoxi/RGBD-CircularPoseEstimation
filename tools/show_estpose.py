from ElpPy.dataproc import GTPoseLoader
import os
import cv2 
import numpy as np
from PIL import Image
import json
from ElpPy.utils import GeneralCamera, GeneralEllipse, GeneralLine, GeneralLineSegment, GeneralSpacialCircle, drawCirclePose, norm_dist, perspectCircular2Image, position_dist, radius_dist
from ElpPy.utils import IoUEllipses
from ElpPy.pose import poseSingleCircleDepthSimple
from tqdm import tqdm


dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingLight1/'
# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark1/'

dataset_name = 'CircularPose-RealRing40Traj'
T_iou_elp = 0.8

usage_method = 'SafaeeDepth'
# usage_method = 'PCL'
# usage_method = 'AprilTagsRGBD'
# usage_method = 'ANEF'
# usage_method = 'PCD'

print(dataset_root_path)
print('eval method {0}'.format(usage_method))

loader = GTPoseLoader()
loader.load(dataset_root_path, dataset_name)
gt_num = len(loader)

total_target_ellipse = 0

all_usage_time = []

total_recall_ellipse = 0
all_usage_ellipse_iou = []
all_usage_norm_diff = []
all_usage_loc_diff = []
all_usage_radius_diff = []

TBC = loader.all_dataset_gt['TBC']


# 1 获取算法在数据集中所有的检测结果
# 2 构建算法邻接矩阵，并记录每个图像标记的真值
# 3 


# 1 加载所有相关的数据
all_estimate_pose = []
all_end_pose = []
all_detelps = []
all_gtelps = []

all_cameras = []

TBC = loader.all_dataset_gt['TBC']
# TBC = np.eye(4)

RBC = TBC[0:3, 0:3]
tBC = TBC[0:3, 3] / 1000 # 转为米
tBC = np.array([tBC]).transpose()

radius = None

for idx_image in range(0, gt_num):
    # idx_image = 14
    print(idx_image)
    
    gt = loader.__getitem__(idx_image, False)
    radius = gt['radius']
    color_full_path = gt['color_full_path']
    gcam = gt['camera']
    
    imgC = cv2.imread(color_full_path)
    
    matches_full_path = os.path.join(dataset_root_path, 'FeaturesMatches', '{0}.npz'.format(idx_image))
    each_image_match_data = np.load(matches_full_path, allow_pickle=True)['data'][()]
    each_gt_match_data = each_image_match_data[0]
    elliptical_pixels = each_gt_match_data['elliptical_pixels'] # N * 2
    elliptical_thinpixels = each_gt_match_data['elliptical_thinpixels'] # N * 2
    if usage_method == 'SafaeeDepth':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'SafaeeDepth', '{0}.npz'.format(idx_image))
        each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
    elif usage_method == 'PCL':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'PCL', '{0}.npz'.format(idx_image))
        each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
    elif usage_method == 'AprilTagsRGBD':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'AprilTagsRGBD', '{0}.npz'.format(idx_image))
        each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
    elif usage_method == 'ANEF':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'ANEF', '{0}.npz'.format(idx_image))
        each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
    elif usage_method == 'PCD':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'PCD', '{0}.npz'.format(idx_image))
        each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
    else:
        assert(0)
        
    each_image_estimate_pose = each_image_estimate_pose[0]
    
    detpose = each_image_estimate_pose['detcircle']
    
    # print(detpose.cnorm, detpose.cloc)
    # exit()
    if detpose is not None:
        drawCirclePose(imgC, gcam, each_image_estimate_pose['detcircle'], elliptical_thinpixels, ellipse_thickness=1)
        # drawCirclePose(imgC, gcam, each_image_estimate_pose['detcircle'], None, ellipse_thickness=1)
    
    imgname = str(idx_image) + '.png'
    cv2.imwrite('./.cache/' + imgname, imgC)    
    # exit()
    
    # cv2.imshow('det', imgC)
    # cv2.waitKey()
        
    # imgT = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
    
    
    
    # img = Image.fromarray(imgT)
    # img.show()
    # exit()