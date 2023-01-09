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


# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingLight3/'
dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark3/'

dataset_name = 'CircularPose-RealRing40Traj'
T_iou_elp = 0.8

# usage_method = 'SafaeeDepth'
# usage_method = 'PCL'
# usage_method = 'AprilTagsRGBD'
# usage_method = 'ANEF'
usage_method = 'PCD'

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

# TBC = np.array([[0.607204754831905,-0.794536564547129,-0.0037461056006643,0.241834442322101],
#                 [-0.794486692478419,-0.607210455771073,0.00929289389736959,1.47096790420335],
#                 [-0.0096582184810628,-0.00266645831228468,-0.999949803148058,127.399965841244],
#                 [0, 0, 0, 1]])


RBC = TBC[0:3, 0:3]
tBC = TBC[0:3, 3] / 1000 # 转为米

# tBC[2] = 0.127

tBC = np.array([tBC]).transpose()

radius = None





for idx_image in range(0, gt_num):
    # print(idx_image)
    gt = loader.__getitem__(idx_image, False)
    radius = gt['radius']
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
    if detpose is not None:
        # print(each_image_estimate_pose['gtelp'])
        # print(each_image_estimate_pose['gtelp'].ellipse_shape_img())
        detelp = perspectCircular2Image(each_image_estimate_pose['camera'], each_image_estimate_pose['detcircle'])
        all_detelps.append(detelp)
        all_gtelps.append(each_image_estimate_pose['gtelp'])
        
        # print(detelp.ellipse_shape_img())
        # print(each_image_estimate_pose['gtelp'].ellipse_shape_img())
        
        all_estimate_pose.append(each_image_estimate_pose['detcircle'])   
        
        all_end_pose.append(each_image_estimate_pose['end_pose'])   
    else:
        all_detelps.append(None)
        all_gtelps.append(None)
        all_estimate_pose.append(None)
        all_end_pose.append(None)
        

    
# 2 统计该数据集中可恢复的椭圆
index_resilient_pose = []
for idx_image in range(gt_num):
    each_detelp = all_detelps[idx_image]
    each_gtelp = all_gtelps[idx_image]
    total_target_ellipse += 1
    if each_detelp is not None:
        # print(each_detelp.ellipse_shape_img(), each_gtelp.ellipse_shape_img())
        iou_elp = IoUEllipses(each_detelp, each_gtelp)
        if iou_elp > T_iou_elp:
            total_recall_ellipse += 1
            # print(iou_elp, total_recall_ellipse, total_target_ellipse)
            index_resilient_pose.append(idx_image)
            
# 3 评估位置法向误差
err_match_norm = []
err_match_pos = []
err_match_radius = []

for each_resilient_idx in index_resilient_pose:
    
    each_detpose = all_estimate_pose[each_resilient_idx]
    each_endpose = all_end_pose[each_resilient_idx]
    err_match_radius.append(abs(each_detpose.cr - radius))
    for cmp_resilient_idx in index_resilient_pose:
        if each_resilient_idx == cmp_resilient_idx:
            continue
        
        
        cmp_det_pose = all_estimate_pose[cmp_resilient_idx]
        
        err_endpose = np.matmul(np.linalg.inv(each_endpose), all_end_pose[cmp_resilient_idx])
        # err_endpose = np.matmul(all_end_pose[cmp_resilient_idx], np.linalg.inv(each_endpose))
        # err_endpose = np.linalg.inv(err_endpose)
        
        Rerr = err_endpose[0:3, 0:3]
        terr = err_endpose[0:3, 3]
        terr = np.array([terr]).transpose()
        Rerrp = np.matmul(np.matmul(RBC.transpose(), Rerr), RBC)
        nz = Rerrp[:, 2]
        cosz = abs(nz[2])/np.linalg.norm(nz)
        # gterr_norm = np.math.acos(cosz) / np.pi * 180 if cosz < 1 else 0
        
        terrp = np.matmul(Rerr, tBC) + terr - tBC
        gterr_t = np.matmul(RBC.transpose(), terrp)
        
        gterr_t = gterr_t[:, 0]
        
        gterr_norm = np.math.acos(cosz) / np.pi * 180 if cosz < 1 else 0 # 偏移角度
        lgterr_t = np.linalg.norm(gterr_t) # 偏移GT长度
        
        # 统计误差
        deterr_norm = norm_dist(cmp_det_pose.cnorm, each_detpose.cnorm)
                    
        deterr_t = cmp_det_pose.cloc - each_detpose.cloc
        ldeterr_t = np.linalg.norm(deterr_t)
        
        err_match_norm.append(abs(deterr_norm - gterr_norm))
        err_match_pos.append(abs(ldeterr_t - lgterr_t))
        
        print(abs(deterr_norm - gterr_norm), abs(ldeterr_t - lgterr_t) * 100, (each_detpose.cr - radius) * 100, each_resilient_idx, cmp_resilient_idx)
    # break



err_match_norm = np.array(err_match_norm)
err_match_pos = np.array(err_match_pos)
err_match_radius = np.array(err_match_radius)
        
# print(err_match_pos * 100)
# print(err_match_norm)


# 输出最终评估结果  
print('Valid Recall: {0} %'.format(total_recall_ellipse * 100.0 / total_target_ellipse))

ratio_norm_05 = len(np.argwhere(err_match_norm < 0.5)) / len(err_match_norm)
ratio_norm_1 = len(np.argwhere(err_match_norm < 1)) / len(err_match_norm)
ratio_norm_5 = len(np.argwhere(err_match_norm < 5)) / len(err_match_norm)
error_norm_1 = np.mean(err_match_norm[err_match_norm < 5])
print('Norm Mean Error: {0}, 0.5: {1}, 1: {2}, 5: {3} %'.format(error_norm_1, 
                                                                ratio_norm_05 * 100, 
                                                                ratio_norm_1 * 100, 
                                                                ratio_norm_5 * 100))   

ratio_loc_05 = len(np.argwhere(err_match_pos < 0.005)) / len(err_match_pos)
ratio_loc_1 = len(np.argwhere(err_match_pos < 0.01)) / len(err_match_pos)
ratio_loc_3 = len(np.argwhere(err_match_pos < 0.03)) / len(err_match_pos)
error_loc_3 = np.mean(err_match_pos[err_match_pos < 0.03])

print('Loc Mean Error: {0}, 0.5cm: {1}, 1cm: {2}, 3cm: {3} %'.format(error_loc_3 * 100, 
                                                                ratio_loc_05 * 100, 
                                                                ratio_loc_1 * 100, 
                                                                ratio_loc_3 * 100))  

ratio_radius_5 = len(np.argwhere(err_match_radius < 0.05)) / len(err_match_radius)
error_radius_5 = np.mean(err_match_radius[err_match_radius < 0.05])

print('Radius Mean Error: {0}, 5cm: {1}'.format(error_radius_5 * 100, ratio_radius_5 * 100))


exit()

for idx_image in range(len(all_gtelps)):
    each_detelp = all_detelps[idx_image]
    each_gtelp = all_gtelps[idx_image]
    each_detpose = all_estimate_pose[idx_image]
    each_endpose = all_end_pose[idx_image]

    total_target_ellipse += 1
    if each_detelp is not None:
        iou_elp = IoUEllipses(each_detelp, each_gtelp)
        if iou_elp > T_iou_elp:
            print(iou_elp)
            total_recall_ellipse += 1
            
            err_match_norm = np.zeros(len(all_end_pose)) + 10000
            err_match_pos = np.zeros(len(all_end_pose)) + 10000
            for idx_pose in range(len(all_end_pose)):
                # idx_pose = 1
                if idx_pose == idx_image:
                    continue
                select_detpose = all_estimate_pose[idx_pose]
                if select_detpose is None:
                    continue
                
                
                # 计算真实位置差异
                # pose_err = np.matmul(all_end_pose[idx_pose], np.linalg.inv(each_endpose))
                # pose_err = np.matmul(each_endpose, np.linalg.inv(all_end_pose[idx_pose]))
                # print(each_endpose,all_end_pose[idx_pose])
                pose_err = np.matmul(np.linalg.inv(each_endpose), all_end_pose[idx_pose])
                # pose_err = np.matmul(np.linalg.inv(all_end_pose[idx_pose]), each_endpose)
                
                # pose_err = np.linalg.inv(pose_err)
                
                # print(idx_image, idx_pose)
                # print('each_endpose', each_endpose)
                # print('all_end_pose[idx_pose]', all_end_pose[idx_pose])
                # print('pose_err', pose_err)
                # print('pose_err-inv', np.linalg.inv(pose_err))
                
                Rerr = pose_err[0:3, 0:3]
                terr = pose_err[0:3, 3]
                terr = np.array([terr]).transpose()
                Rerrp = np.matmul(np.matmul(RBC.transpose(), Rerr), RBC)
                
                nz = Rerrp[:, 2]
                cosz = abs(nz[2])/np.linalg.norm(nz)
                gterr_norm = np.math.acos(cosz) / np.pi * 180 if cosz < 1 else 0
                
                terrp = np.matmul(Rerr, tBC) + terr - tBC
                gterr_t = np.matmul(RBC.transpose(), terrp)
                
                gterr_t = gterr_t[:, 0]
                lgterr_t = np.linalg.norm(gterr_t) # 偏移GT长度
                # print('tBC', tBC)
                # print('terr', terr)
                # 
                if select_detpose is not None:
                    deterr_norm = norm_dist(select_detpose.cnorm, each_detpose.cnorm)
                    
                    deterr_t = select_detpose.cloc - each_detpose.cloc
                    ldeterr_t = np.linalg.norm(deterr_t)
                    
                    err_match_norm[idx_pose] = abs(deterr_norm - gterr_norm)
                    err_match_pos[idx_pose] = abs(ldeterr_t - lgterr_t)
                
                    print(err_match_norm[idx_pose], err_match_pos[idx_pose] )
                    
                    # print('lgterr_norm', gterr_norm)
                    # print('ldeterr_t', deterr_norm)
                    
                    # print('lgterr_t', lgterr_t)
                    # print('ldeterr_t', ldeterr_t)

                
            # 排序结果，并获得整体评估结果
            print('idx_image', idx_image)
            print('err_match_norm', err_match_norm)
            print('err_match_pos', err_match_pos) 
            exit()
                
            
    

print('Valid Recall: {0} %'.format(total_recall_ellipse * 100.0 / total_target_ellipse))



exit()
for idx_image in range(0, gt_num):
    # print('Processing {0}th image.'.format(idx_image))
    gt = loader.__getitem__(idx_image, False)
    
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
    
    
    # 需要标记的椭圆，和末端变化量
    
    
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


    
    

