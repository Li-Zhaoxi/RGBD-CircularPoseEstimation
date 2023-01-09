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
from scipy.io import savemat
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='record_error_distance')
    parser.add_argument('--ds_root', type=str, 
                        default='/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/',
                        help='The root path of the used dataset')
    parser.add_argument('--ds_name', type=str,
                        default='CircularPose-15PlaneBlender',
                        help='The dsname is used to generate usage datas in class GTPoseLoader')
    parser.add_argument('--method', type=str,
                        default='SafaeeDepth',
                        help='The method of pose estimation')
    parser.add_argument('--noise', type=str,
                        default='nonoise',help='noise type')
    
    args = parser.parse_args()
    return args

opt = parse_arguments()

dataset_root_path = opt.ds_root
dataset_name = opt.ds_name
usage_method = opt.method
noise_type = opt.noise

print('dataset_root_path:', dataset_root_path)
print('dataset_name:', dataset_name)
print('usage_method:', usage_method)
print('noise_type:', noise_type)


# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlenderNoise/'

# dataset_name = 'CircularPose-15PlaneBlender'
T_iou_elp = 0.8

# usage_method = 'SafaeeDepth'
# usage_method = 'PCL'
# usage_method = 'AprilTagsRGBD'
# usage_method = 'PCD'


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

all_centers = []
all_iou = []
all_norm_diff = []
all_loc_diff = []
all_radius_diff = []

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
            
            all_centers.append(gtcircle.cloc)
            all_iou.append(iou_elp)
            all_norm_diff.append(norm_dist(gtcircle.cnorm, det_poes.cnorm))
            all_loc_diff.append(position_dist(gtcircle.cloc, det_poes.cloc))
            all_radius_diff.append(radius_dist(gtcircle.cr, det_poes.cr))
        else:
            all_usage_time.append(-1)
                        
            all_centers.append(gtcircle.cloc)
            all_iou.append(-1)
            all_norm_diff.append(-1)
            all_loc_diff.append(-1)
            all_radius_diff.append(-1)
            
print('total_target_ellipse', total_target_ellipse)
            

                
            
savemat('sync_z_analy_{0}_{1}.mat'.format(usage_method, noise_type), {'all_centers': all_centers, 'all_iou': all_iou, 
                             'all_norm_diff': all_norm_diff, 'all_loc_diff': all_loc_diff, 
                             'all_radius_diff': all_radius_diff})