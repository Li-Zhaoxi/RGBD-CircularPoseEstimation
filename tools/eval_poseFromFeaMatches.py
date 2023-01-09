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

# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlenderNoise/'

dataset_name = 'CircularPose-15PlaneBlender'
T_iou_elp = 0.8

# usage_method = 'SafaeeDepth'
# usage_method = 'PCL'
# usage_method = 'AprilTagsRGBD'
# usage_method = 'ANEF'
usage_method = 'PCD'


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



for idx_image in tqdm(range(0, gt_num)):
    # print('Processing {0}th image.'.format(idx_image))
    
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
        
    usage_keys = list(each_image_estimate_pose.keys())
    
    for each_key in usage_keys:
        each_gt_estimate_pose = each_image_estimate_pose[each_key]
        
        gtcircle = each_gt_estimate_pose['gtcircle']
        det_poes = each_gt_estimate_pose['detcircle']
        gcam = each_gt_estimate_pose['camera']
        usage_time = each_gt_estimate_pose['time']
    
    
        total_target_ellipse += 1
        
        if det_poes is not None:
            all_usage_time.append(usage_time)
            
            gelp_det = perspectCircular2Image(gcam, det_poes)
            gelp_gt = perspectCircular2Image(gcam, gtcircle)
            
            iou_elp = IoUEllipses(gelp_det, gelp_gt)
            if iou_elp >= T_iou_elp:
                
                total_recall_ellipse += 1
                all_usage_ellipse_iou.append(iou_elp)
                
                angle_diff = norm_dist(gtcircle.cnorm, det_poes.cnorm)
                position_diff = position_dist(gtcircle.cloc, det_poes.cloc)
                radius_diff = radius_dist(gtcircle.cr, det_poes.cr)
                
                all_usage_norm_diff.append(angle_diff)
                all_usage_loc_diff.append(position_diff)
                all_usage_radius_diff.append(radius_diff)
                
                # if angle_diff > 2 or position_diff * 100 > 2 or radius_diff * 100 > 2:
                #     print('idx_gt: {3}, angle_diff: {0}, position_diff: {1} cm, radius_diff: {2} cm'.format(
                #         angle_diff, position_diff * 100, radius_diff * 100, each_key
                #     ))
                
            

all_usage_ellipse_iou = np.array(all_usage_ellipse_iou)
all_usage_norm_diff = np.array(all_usage_norm_diff)
all_usage_loc_diff = np.array(all_usage_loc_diff)
all_usage_radius_diff = np.array(all_usage_radius_diff)
all_usage_time = np.array(all_usage_time)
            
print('Valid Recall: {0} %'.format(total_recall_ellipse * 100.0 / total_target_ellipse))


ratio_norm_05 = len(np.argwhere(all_usage_norm_diff < 0.5)) / len(all_usage_norm_diff)
ratio_norm_1 = len(np.argwhere(all_usage_norm_diff < 1)) / len(all_usage_norm_diff)
ratio_norm_5 = len(np.argwhere(all_usage_norm_diff < 5)) / len(all_usage_norm_diff)
error_norm_1 = np.mean(all_usage_norm_diff[all_usage_norm_diff < 1])

print('Norm Mean Error: {0}, 0.5: {1}, 1: {2}, 5: {3} %'.format(error_norm_1, 
                                                                ratio_norm_05 * 100, 
                                                                ratio_norm_1 * 100, 
                                                                ratio_norm_5 * 100))   

ratio_loc_1 = len(np.argwhere(all_usage_loc_diff < 0.01)) / len(all_usage_loc_diff)
ratio_loc_5 = len(np.argwhere(all_usage_loc_diff < 0.05)) / len(all_usage_loc_diff)
ratio_loc_10 = len(np.argwhere(all_usage_loc_diff < 0.1)) / len(all_usage_loc_diff)
error_loc_5 = np.mean(all_usage_loc_diff[all_usage_loc_diff < 0.05])

print('Loc Mean Error: {0}, 1cm: {1}, 5cm: {2}, 10cm: {3} %'.format(error_loc_5 * 100, 
                                                                ratio_loc_1 * 100, 
                                                                ratio_loc_5 * 100, 
                                                                ratio_loc_10 * 100))  

ratio_radius_5 = len(np.argwhere(all_usage_radius_diff < 0.05)) / len(all_usage_radius_diff)
error_radius_5 = np.mean(all_usage_radius_diff[all_usage_radius_diff < 0.05])

print('Radius Mean Error: {0}, 5cm: {1}'.format(error_radius_5 * 100, ratio_radius_5 * 100))
        
print('IoU Average: {0} %'.format(np.mean(all_usage_ellipse_iou * 100)))

print('Average Time: {0}ms'.format(np.mean(all_usage_time)))


    
    

