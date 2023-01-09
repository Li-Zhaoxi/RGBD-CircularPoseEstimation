from ast import parse
from ElpPy.dataproc import GTPoseLoader
import os
import cv2 
import numpy as np
from PIL import Image
import json
from ElpPy.utils import GeneralCamera, GeneralEllipse, GeneralLine, GeneralLineSegment, GeneralSpacialCircle, drawCirclePose, norm_dist, perspectCircular2Image, position_dist, radius_dist
from ElpPy.pose import PerspectiveCircleDepth, poseANEFDepth, posePCL_LineParallelY, poseSingleCircleDepthSimple, poseSquareDepth
from tqdm import tqdm
import argparse




def parse_arguments():
    parser = argparse.ArgumentParser(description='start_calPoseFromFeaMatches')
    parser.add_argument('--ds_root', type=str, 
                        default='/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/',
                        help='The root path of the used dataset')
    parser.add_argument('--ds_name', type=str,
                        default='CircularPose-15PlaneBlender',
                        help='The dsname is used to generate usage datas in class GTPoseLoader')
    parser.add_argument('--method', type=str,
                        default='SafaeeDepth',
                        help='The method of pose estimation')
    
    args = parser.parse_args()
    return args

opt = parse_arguments()
dataset_root_path = opt.ds_root
dataset_name = opt.ds_name
usage_method = opt.method
# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlenderNoise/'

# dataset_name = 'CircularPose-15PlaneBlender'
loader = GTPoseLoader()
loader.load(dataset_root_path, dataset_name)


# usage_method = 'SafaeeDepth'
# usage_method = 'PCL'
# usage_method = 'AprilTagsRGBD'
# usage_method = 'ANEF'
# usage_method = 'PCD'

gt_num = len(loader)
print('Process method: {0}, dataset number: {1}, dataset_root_path: {2}, dataset_name: {3}.'.format(usage_method, gt_num, dataset_root_path, dataset_name))
for idx_image in tqdm(range(0, gt_num)):
    # idx_image = 13
    # print('Processing {0}th image.'.format(idx_image))
    gt = loader.__getitem__(idx_image, True)
    
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
    
    total_gt = 1 # 理论真值个数
    imgC = gt['imgC']
    imgG = gt['imgG']
    imgD = gt['imgD']
    
    radius = gt['radius']
    # print('radius', radius)
    # exit()
    
    
    each_image_estimate_pose = {}
    for idx_gt in range(total_gt):
        if idx_gt not in usage_keys:
            continue
        # print(idx_gt)
        # if idx_gt != 12:
        #     continue
        each_gt_match_data = each_image_match_data[idx_gt]
        

        elliptical_pixels = each_gt_match_data['elliptical_pixels'] # N * 2
        elliptical_thinpixels = each_gt_match_data['elliptical_thinpixels']
        
        gelp = each_gt_match_data['ellipse']
        
        end_pose = each_gt_match_data['end_pose']
        joint_angle = each_gt_match_data['joint_angle']
        
        # res_gt['end_pose'] = usage_gt['end_pose']
        # res_gt['joint_angle'] = usage_gt['joint_angle']
        
        # print(each_gt_match_data['ellipse'])
        # gelp.loadData(each_gt_match_data['ellipse'])
        
        mask_pts = each_gt_match_data['mask_pts'] # N * 2
        # print(each_gt_match_data)
        square_pixels = each_gt_match_data['square_pixels'] # 4 * 2
        square_pixels_W = np.array(each_gt_match_data['square_pixels_W'])
        cooperate_line = each_gt_match_data['cooperate_line'] # 2 * 2
        cooperate_line_W = each_gt_match_data['cooperate_line_W']
        t1 = cv2.getTickCount()
        if usage_method == 'SafaeeDepth':
            # print(gelp, gtcircle, imgD, mask_pts)
            # print(gelp.ellipse_shape_img())
            shape_img = gelp.ellipse_shape_img()
            shape_img[0] += 1
            shape_img[1] += 1
            
            usage_elp = GeneralEllipse(elp_data=shape_img, elp_type='shape_image')
            if mask_pts is None:
                det_pose = None
            else:
                det_pose = poseSingleCircleDepthSimple(gcam, usage_elp, radius, imgD, 1, mask_pts[:, [1, 0]])
        elif usage_method == 'PCL':
            if cooperate_line is None:
                det_pose = None
            else:
                shape_img = gelp.ellipse_shape_img()
                shape_img[0] += 1
                shape_img[1] += 1
                usage_elp = GeneralEllipse(elp_data=shape_img, elp_type='shape_image')
                
                # print(cooperate_line_W)
                cooperate_line_W[0, 2] *= -1
                cooperate_line_W[1, 2] *= -1
                # print(cooperate_line_W)
                
                pcl_pose = posePCL_LineParallelY(gcam, usage_elp, radius, cooperate_line, cooperate_line_W)
                det_pose = pcl_pose['final']
                
                # cooperate_line = pcl_pose['line_Ipts']
                
        elif usage_method == 'AprilTagsRGBD':
            if square_pixels is None:
                det_pose = None
            else:
                cor1 = square_pixels_W[0, :]
                cor2 = square_pixels_W[1, :]
                square_length = np.linalg.norm(cor1 - cor2)

                cnorm, cloc = poseSquareDepth(gcam, square_length, square_pixels, imgD,  1)
                # print(cnorm, cloc)
                det_pose = GeneralSpacialCircle(cnorm.transpose()[0], cloc.transpose()[0], radius)
        elif usage_method == 'ANEF':
            if mask_pts is None:
                det_pose = None
            else:
                # mask_pts是按照行列标寸的，elliptical_pixels是按照图像xy存的
                det_pose = poseANEFDepth(gcam, elliptical_pixels + 1, radius, imgD, 1, mask_pts[:, [1, 0]], tolMax=2.0)
                # det_pose = poseANEFDepth(gcam, elliptical_thinpixels, radius, imgD, 1, mask_pts[:, [1, 0]], tolMax=3.0)
        elif usage_method == 'PCD':
            pcd = PerspectiveCircleDepth(gcam, 1)
            if mask_pts is None:
                shape_img = gelp.ellipse_shape_img()
                shape_img[0] += 1
                shape_img[1] += 1
                usage_elp = GeneralEllipse(elp_data=shape_img, elp_type='shape_image')
                
                # center = usage_elp.get_center_img()
                
                # print(usage_elp.ellipse_shape_mat())
                
                pcd_pose = pcd(imgD, None, elliptical_thinpixels[:, [1, 0]], usage_elp.ellipse_shape_mat())
                
                
            
            else:
                # print(gtcircle.cnorm, gtcircle.cloc, gtcircle.cr)
                
                # imgC[elliptical_pixels[:, 1], elliptical_pixels[:, 0]] = [0, 255, 0]
                # cv2.imwrite('test.png', imgC)
                
                pcd_pose = pcd(imgD, mask_pts[:, [1, 0]], elliptical_thinpixels[:, [1, 0]])
                # pcd_pose = pcd(imgD, mask_pts[:, [1, 0]], elliptical_pixels[:, [1, 0]] + 1)
                
                
                # det_pose = pcd_pose['medium']
                
                
                # support_csp = pcd_pose['support_csp']
                # uvs = gcam.project_point_to_pixel(support_csp)
                # for idx in range(uvs.shape[0]):
                #     pt = uvs[idx, :]
                #     u = int(pt[0])
                #     v = int(pt[1])
                #     imgC[v, u, 0] = 255
                #     imgC[v, u, 1] = 255
                #     imgC[v, u, 2] = 0
            # det_pose = pcd_pose['initial']
            det_pose = pcd_pose['final']
                
        else:
            assert(0)
        t2 = cv2.getTickCount()
        usage_time = (t2 - t1) * 1000.0 / cv2.getTickFrequency()
        
        each_gt_estimate_pose = {}
        each_gt_estimate_pose['detcircle'] = det_pose
        each_gt_estimate_pose['gtelp'] = each_gt_match_data['gtellipse']
        
        # print(gelp.ellipse_shape_img())
        
        each_gt_estimate_pose['camera'] = gcam
        each_gt_estimate_pose['time'] = usage_time
        
        each_gt_estimate_pose['end_pose'] = end_pose
        each_gt_estimate_pose['joint_angle'] = joint_angle
        
        each_image_estimate_pose[idx_gt] = each_gt_estimate_pose
        
    #     if det_pose is not None:
    #         # if norm_dist(gtcircle.cnorm, det_pose.cnorm) <= 1:
    #         #     continue
    #         # print('gtcircle.cnorm', gtcircle.cnorm)
    #         # print('idx_Gt: ', idx_gt)
    #         # print('norm dist: ', norm_dist(gtcircle.cnorm, det_pose.cnorm))
        
    #         # print('position dist: ', 100 * position_dist(gtcircle.cloc, det_pose.cloc), 100 * gtcircle.cloc, 100 * det_pose.cloc)
    #         # print('radius dist: ', 100 * radius_dist(gtcircle.cr, det_pose.cr))
            
    #         print(det_pose.cnorm, det_pose.cloc)
        
    #         drawCirclePose(imgC, gcam, det_pose, 
    #                     elliptical_pixels=elliptical_pixels, square_pixels=square_pixels, other_pts=None,
    #                     mask_pts=None, cooperate_line=cooperate_line, ingelp=None)
            
    #         # drawCirclePose(imgC, gcam, det_pose, 
    #         #             elliptical_pixels=None, square_pixels=square_pixels, other_pts=None,
    #         #             mask_pts=mask_pts, cooperate_line=cooperate_line, ingelp=None)
            
    #     # break
        
    
    # imgT = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
    # show_image = Image.fromarray(imgT)
    # show_image.show()
    # exit()
    
    if usage_method == 'SafaeeDepth':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'SafaeeDepth', '{0}'.format(idx_image))
        np.savez(save_npz_path, data = each_image_estimate_pose)
    elif usage_method == 'PCL':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'PCL', '{0}'.format(idx_image))
        np.savez(save_npz_path, data = each_image_estimate_pose)
    elif usage_method == 'AprilTagsRGBD':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'AprilTagsRGBD', '{0}'.format(idx_image))
        np.savez(save_npz_path, data = each_image_estimate_pose)
    elif usage_method == 'ANEF':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'ANEF', '{0}'.format(idx_image))
        np.savez(save_npz_path, data = each_image_estimate_pose)
    elif usage_method == 'PCD':
        save_npz_path = os.path.join(dataset_root_path, 'Methods', 'PCD', '{0}'.format(idx_image))
        np.savez(save_npz_path, data = each_image_estimate_pose)
    else:
        assert(0)
        
        
    # exit()



        
        
    
    
