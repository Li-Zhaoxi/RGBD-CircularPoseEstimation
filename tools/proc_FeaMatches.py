from ElpPy.dataproc import GTPoseLoader
import os
import cv2 
import numpy as np
from PIL import Image
import json
from ElpPy.utils import GeneralEllipse, GeneralLine, GeneralLineSegment, drawCirclePose, perspectCircular2Image
from tools.features_load import loadELSDJson, loadGTMask, loadHarrisJson, loadLSDJson, loadMaskRCNN
from tqdm import tqdm


# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlenderNoise/'
dataset_name = 'CircularPose-15PlaneBlender'



loader = GTPoseLoader()
loader.load(dataset_root_path, dataset_name)

gt_num = len(loader)
print(gt_num)

# ELSD算法存在1个像素的偏移
ellipse_pixel_offset = 1

# 记录每一张图
#  记录存在匹配的结果，分为features和gt
# all_match_data = {}
for idx_image in tqdm(range(0, gt_num)):
    # idx_image = 570
    # print('Processing {0}th image.'.format(idx_image))
    gt = loader.__getitem__(idx_image, False)
    
    color_full_path = gt['color_full_path']
    root_path = gt['root_path']
    
    # print(color_full_path)
    
    color_name = os.path.basename(color_full_path)
    
    usage_match_data = {}
    
    # 加载真值信息
    usage_gscir = gt['spacialcircle']
    usage_innersquare = gt['innersquare']
    usage_innersquare_W = gt['innersquare_W']
    usage_leftlines = gt['leftline'] # K * 2 * 2
    usage_leftlines_W = gt['leftline_W'] # K * 2 * 2
    usage_gcam = gt['camera']
    
    total_gt = len(usage_gscir) # 理论真值个数
    
    # 加载检测信息
    ELSD_Json_path = os.path.join(root_path, 'ExtractedFeatures/ELSD', color_name + '.json')
    det_elps = loadELSDJson(ELSD_Json_path)
    # print(det_elps)
    
    mask_data_path = os.path.join(root_path, 'ExtractedFeatures/MaskRCNN', color_name + '.npz')
    det_masks = loadMaskRCNN(mask_data_path)
    mask_gt_path = os.path.join(root_path, 'mask', 'MASK_{0}.png'.format(idx_image))
    gt_masks = loadGTMask(mask_gt_path)
    
    LSD_Json_path = os.path.join(root_path, 'ExtractedFeatures/LSD', color_name + '.json')
    det_lsd = loadLSDJson(LSD_Json_path)
    
    Harris_Json_path = os.path.join(root_path, 'ExtractedFeatures/Harris', color_name + '.json')
    det_corners = loadHarrisJson(Harris_Json_path)
    
    # 设置匹配阈值
    T_elp = 6
    T_corner = 6
    T_line = 6
    
    imgC = cv2.imread(color_full_path)
    
    each_image_match_data = {}
    # 开始匹配，存在匹配椭圆则进行其他特征的匹配
    for idx_gt in range(total_gt):
        # if idx_gt != 5:
        #     continue
        
        is_find = False
        gelp = perspectCircular2Image(usage_gcam, usage_gscir[idx_gt])
        found_ellipticl_pts = None
        found_elp_data = None
        found_ellipse = None
        
        for each_det_elp in det_elps:
            tmp_pts = np.array(each_det_elp.regs)
            dist = gelp.calDistance(tmp_pts + ellipse_pixel_offset)
            # print(np.mean(dist), np.max(dist), np.min(dist))
            # print(np.min(dist))
            if np.mean(dist) < T_elp:
                if found_ellipticl_pts is None or len(found_ellipticl_pts) < len(tmp_pts):
                    found_ellipticl_pts = tmp_pts
                    is_find = True
                    found_elp_data = each_det_elp
        if not is_find:
            continue
        cx = found_elp_data.cx
        cy = found_elp_data.cy
        rx = found_elp_data.rx
        ry = found_elp_data.ry
        angle = found_elp_data.angle /180.0 * np.pi
        # print('angle', angle)
        elp_data = np.array([cx, cy, rx, ry, angle])
        elp_type = 'shape_image'
        found_ellipse = GeneralEllipse(elp_data=elp_data, elp_type=elp_type)
        
        # 寻找Mask
        found_mask = None
        num_masks = det_masks.shape[0]
        for idx_masks in range(num_masks):
            usage_mask = det_masks[idx_masks, 0]
            # tmpimage = Image.fromarray(usage_mask)
            # tmpimage.show()
            # print(usage_mask.shape)
            # print(np.argwhere(usage_mask > 0))
            idxes = np.argwhere(usage_mask > 0)
            mask_px = idxes[:, 0]
            mask_py = idxes[:, 1]
            # mask_px, mask_py = np.argwhere(usage_mask > 0)
            idx = np.argwhere(gt_masks[mask_px, mask_py] == idx_gt + 1)
            if len(idx) > 3:
                if found_mask is None:
                    found_mask = np.vstack([mask_py, mask_px]).transpose()
                else:
                    if found_mask.shape[0] < len(idx):
                        found_mask = np.vstack([mask_py, mask_px]).transpose()
        # if pts_mask is not None:
        #     print('find', idx_gt)
        # exit()
        
        # 寻找直线段
        found_linesements = None
        usage_s_range = None
        usage_lline = usage_leftlines[idx_gt]
        # print(usage_lline)
        # exit()
        usage_lline_W = usage_leftlines_W[idx_gt]
        if len(usage_lline) > 0:
            glinesegment = GeneralLineSegment(pt1=usage_lline[0], pt2=usage_lline[1])
            for each_line in det_lsd:
                pt1 = each_line[:2]
                pt2 = each_line[2:]
                tpts = np.vstack([pt1, pt2])
                dist = glinesegment.calDistance(tpts)
                sper = glinesegment.calPerspectiveScale(tpts)
                if np.mean(dist) < T_line:
                    s1, s2 = np.sort(sper)
                    smin = np.max([0.0, s1])
                    smax = np.min([1.0, s2])
                    if smin < 1: # 有交集
                        if usage_s_range is None:
                            found_linesements = tpts
                            usage_s_range = [smin, smax]
                        else:
                            if usage_s_range[1] - usage_s_range[0] < smax - smin:
                                found_linesements = tpts
                                usage_s_range = [smin, smax]
                                                
            
        
        # 寻找方形角点
        usage_square = usage_innersquare[idx_gt]
        usage_square_W = usage_innersquare_W[idx_gt]
        found_corners = []
        for each_gt_pt in usage_square:
            dist = np.linalg.norm(det_corners - each_gt_pt, axis=1)
            # print(np.min(dist))
            idx = np.argmin(dist)
            if dist[idx] < T_corner:
                found_corners.append(det_corners[idx])
        found_corners = np.array(found_corners)
        if len(found_corners) != 4:
            found_corners = None
            
        each_gt_match_data = {}
        each_gt_match_data['gtcircle'] = usage_gscir[idx_gt].getDict()
        each_gt_match_data['elliptical_pixels'] = found_ellipticl_pts
        each_gt_match_data['ellipse'] = found_ellipse
        each_gt_match_data['mask_pts'] = found_mask
        each_gt_match_data['square_pixels'] = found_corners
        each_gt_match_data['square_pixels_W'] = usage_square_W
        each_gt_match_data['cooperate_line'] = found_linesements
        each_gt_match_data['cooperate_line_W'] = usage_lline_W
        each_image_match_data[idx_gt] = each_gt_match_data
        # print('found_linesements', found_linesements)
        # drawCirclePose(imgC, usage_gcam, usage_gscir[idx_gt], 
        #                elliptical_pixels=found_ellipticl_pts, square_pixels=found_corners, other_pts=None,
        #                mask_pts=found_mask, cooperate_line=found_linesements, ingelp=found_ellipse)
    
    each_image_match_data['camera'] = usage_gcam.getDict()
    
    save_npz_path = os.path.join(dataset_root_path, 'FeaturesMatches', '{0}'.format(idx_image))
    np.savez(save_npz_path, data=each_image_match_data)
    
    # all_match_data[idx_image] = each_image_match_data
    
    # imgT = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
    # show_image = Image.fromarray(imgT)
    # show_image.show()
    # exit()
            
            
# save_npz_path = os.path.join(dataset_root_path, 'features_matches')
# np.savez(save_npz_path, data=all_match_data)
    