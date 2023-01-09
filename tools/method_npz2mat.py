from ElpPy.dataproc import GTPoseLoader
import os
import cv2 
import numpy as np
from PIL import Image
import json
from ElpPy.utils import GeneralCamera, GeneralEllipse, GeneralLine, GeneralLineSegment, GeneralSpacialCircle, depth_colorizer, drawCirclePose, norm_dist, perspectCircular2Image, position_dist, radius_dist
from ElpPy.utils import IoUEllipses
from ElpPy.pose import poseSingleCircleDepthSimple
from tqdm import tqdm
from scipy.io import savemat



# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlenderNoise/'

dataset_name = 'CircularPose-15PlaneBlender'
T_iou_elp = 0.8



usage_methods = ['SafaeeDepth', 'PCL', 'AprilTagsRGBD', 'ANEF', 'PCD']



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
    idx_image = 1281
    # print('Processing {0}th image.'.format(idx_image))
    gt = loader.__getitem__(idx_image, True)
    
    # 获取匹配出的图像合作信息
    color_full_path = gt['color_full_path']
    matches_full_path = os.path.join(dataset_root_path, 'FeaturesMatches', '{0}.npz'.format(idx_image))
    each_image_match_data = np.load(matches_full_path, allow_pickle=True)['data'][()]
    
    usage_full_Rot = gt['full_rot']
    
    camera_dict = each_image_match_data['camera']
    gcam = GeneralCamera.loadDict(camera_dict)
    usage_gscir = gt['spacialcircle']
    total_gt = len(usage_gscir) # 理论真值个数
    usage_keys = list(each_image_match_data.keys())
    
    save_mat_dict = {}
    
    save_mat_dict['usage_keys'] = usage_keys
    save_mat_dict['camera'] = camera_dict
    imgC = gt['imgC']
    imgD = gt['imgD']
    
    imgDC = depth_colorizer(imgD)
    
    # tmpdept = Image.fromarray(imgDC)
    # tmpdept.show()
    # exit()
    
    
    save_mat_dict['imgC'] = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
    save_mat_dict['imgDC'] = cv2.cvtColor(imgDC, cv2.COLOR_BGR2RGB)
    save_mat_dict['preshapes'] = []
    for idx_gt in range(total_gt):
        if idx_gt not in usage_keys:
            continue
        
        tmp = {}
        each_gt_match_data = each_image_match_data[idx_gt]
        # 椭圆像素点
        elliptical_pixels = each_gt_match_data['elliptical_pixels'] # N * 2
        # 椭圆形状
        gelp = each_gt_match_data['ellipse']
        # 目标平面mask
        mask_pts = each_gt_match_data['mask_pts'] # N * 2
        # 方形像素点
        square_pixels = each_gt_match_data['square_pixels'] # 4 * 2
        # 合作直线
        cooperate_line = each_gt_match_data['cooperate_line'] # 2 * 2
        
        tmp['elliptical_pixels'] = elliptical_pixels
        tmp['ellipse'] = gelp.getDict()
        if mask_pts is not None:
            tmp['mask_pts'] = mask_pts
        else:
            tmp['mask_pts'] = []
        if square_pixels is not None:
            tmp['square_pixels'] = square_pixels
        else:
            tmp['square_pixels'] = []
        if cooperate_line is not None:
            tmp['cooperate_line'] = cooperate_line
        else:
            tmp['cooperate_line'] = []
        tmp['gtidx'] = idx_gt
        tmp['Rot'] = usage_full_Rot[idx_gt]
        save_mat_dict['preshapes'].append(tmp)
        
        
        
    usage_keys = []
    image_estimate_poses = []
    for each_method in usage_methods:
        if each_method == 'SafaeeDepth':
            save_npz_path = os.path.join(dataset_root_path, 'Methods', 'SafaeeDepth', '{0}.npz'.format(idx_image))
            each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
        elif each_method == 'PCL':
            save_npz_path = os.path.join(dataset_root_path, 'Methods', 'PCL', '{0}.npz'.format(idx_image))
            each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
        elif each_method == 'AprilTagsRGBD':
            save_npz_path = os.path.join(dataset_root_path, 'Methods', 'AprilTagsRGBD', '{0}.npz'.format(idx_image))
            each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
        elif each_method == 'ANEF':
            save_npz_path = os.path.join(dataset_root_path, 'Methods', 'ANEF', '{0}.npz'.format(idx_image))
            each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
        elif each_method == 'PCD':
            save_npz_path = os.path.join(dataset_root_path, 'Methods', 'PCD', '{0}.npz'.format(idx_image))
            each_image_estimate_pose = np.load(save_npz_path, allow_pickle=True)['data'][()]
        else:
            assert(0)
        # print(image_estimate_poses)
        usage_keys.extend(list(each_image_estimate_pose.keys()))
        image_estimate_poses.append(each_image_estimate_pose)
    
    usage_keys = sorted(list(set(usage_keys)))
    
    pose_result = []
    for each_key in usage_keys:
        print('Process gt index: {0}'.format(each_key))
        
        each_key_result = []
        for method_idx, each_method_pose in enumerate(image_estimate_poses):
            if each_key not in list(each_method_pose.keys()):
                print('no pose')
                continue
            
            each_estimate_pose = each_method_pose[each_key]
            
            gtcircle = each_estimate_pose['gtcircle']
            det_pose = each_estimate_pose['detcircle']
            
            if det_pose is None:
                print('no pose')
                continue
            
            current_norm_diff = norm_dist(gtcircle.cnorm, det_pose.cnorm)
            current_loc_diff = position_dist(gtcircle.cloc, det_pose.cloc)
            
            print('Dnorm: {0}, Dloc: {1}, gtloc: {2}, detloc: {3}'.format(
                current_norm_diff, current_loc_diff, gtcircle.cloc, det_pose.cloc
            ))
            
            each_key_each_method = {}
            each_key_each_method['key'] = each_key
            each_key_each_method['idxmethod'] = method_idx
            each_key_each_method['gtcircle'] = gtcircle.getDict()
            each_key_each_method['det_pose'] = det_pose.getDict()
            each_key_result.append(each_key_each_method)
        pose_result.append(each_key_result)
            
    save_mat_dict['poses'] = pose_result
    
    savemat('showPoseResult.mat', save_mat_dict)
    # print(usage_keys)
    
    exit()
        
        
#     usage_keys = list(each_image_estimate_pose.keys())
    
#     for each_key in usage_keys:
#         each_gt_estimate_pose = each_image_estimate_pose[each_key]
        
#         gtcircle = each_gt_estimate_pose['gtcircle']
#         det_poes = each_gt_estimate_pose['detcircle']
#         gcam = each_gt_estimate_pose['camera']
#         usage_time = each_gt_estimate_pose['time']
    
    
#         total_target_ellipse += 1
        
#         if det_poes is not None:
#             all_usage_time.append(usage_time)
            
#             gelp_det = perspectCircular2Image(gcam, det_poes)
#             gelp_gt = perspectCircular2Image(gcam, gtcircle)
            
#             iou_elp = IoUEllipses(gelp_det, gelp_gt)
            
#             all_centers.append(gtcircle.cloc)
#             all_iou.append(iou_elp)
#             all_norm_diff.append(norm_dist(gtcircle.cnorm, det_poes.cnorm))
#             all_loc_diff.append(position_dist(gtcircle.cloc, det_poes.cloc))
#             all_radius_diff.append(radius_dist(gtcircle.cr, det_poes.cr))
#         else:
#             all_usage_time.append(-1)
                        
#             all_centers.append(gtcircle.cloc)
#             all_iou.append(-1)
#             all_norm_diff.append(-1)
#             all_loc_diff.append(-1)
#             all_radius_diff.append(-1)
            
# print('total_target_ellipse', total_target_ellipse)
            

                
            
# savemat('sync_z_analy_{0}_noise.mat'.format(usage_method), {'all_centers': all_centers, 'all_iou': all_iou, 
#                              'all_norm_diff': all_norm_diff, 'all_loc_diff': all_loc_diff, 
#                              'all_radius_diff': all_radius_diff})