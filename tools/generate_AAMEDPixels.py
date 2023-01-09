from cv2 import cvtColor
from ElpPy.dataproc import GTPoseLoader
from ElpPy.utils import calCannyThreshold
import os
import numpy as np
from PIL import Image
import cv2


# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingLight3/'

dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark3/'
dataset_name = 'CircularPose-RealRing40Traj'

print(dataset_root_path)


loader = GTPoseLoader()
loader.load(dataset_root_path, dataset_name)
gt_num = len(loader)


for idx_image in range(gt_num):
    # idx_image = 13
    # print(idx_image)
    gt = loader.__getitem__(idx_image, True)
    
    imgC = gt['imgC']
    imgG = gt['imgG']
    matches_full_path = os.path.join(dataset_root_path, 'FeaturesMatches', '{0}.npz'.format(idx_image))
    each_image_match_data = np.load(matches_full_path, allow_pickle=True)['data'][()]
    
    total_gt = 1 # 理论真值个数
    usage_keys = list(each_image_match_data.keys())
    
    
    elp_pts_mask = np.zeros_like(imgG, dtype=np.uint8)
    
    each_image_estimate_pose = {}
    for idx_gt in range(total_gt):
        if idx_gt not in usage_keys:
            continue
        each_gt_match_data = each_image_match_data[idx_gt]
        elliptical_pixels = each_gt_match_data['elliptical_pixels'] + 1# N * 2
        gelp = each_gt_match_data['ellipse']
        
        
        for idx_pt in range(elliptical_pixels.shape[0]):
            pt = elliptical_pixels[idx_pt, :]
            elp_pts_mask[pt[1], pt[0]] = 255
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    elp_pts_mask = cv2.morphologyEx(elp_pts_mask, cv2.MORPH_CLOSE, kernel)
    
    low, high = calCannyThreshold(imgG)
    # print(low, high)
    imgGt = cv2.GaussianBlur(imgG, (5, 5), 0.8)
    img_edge = cv2.Canny(imgGt, low, high)
    
    img_edge_mask = np.bitwise_and(img_edge, elp_pts_mask)
    
    # contours = cv2.findContours(img_edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours = contours[0]
    # usage_contours = []
    # for each_contour in contours:
    #     if len(each_contour) < 30:
    #         continue
    #     usage_contours.append(np.squeeze(np.array(each_contour))) 
    
    
    elp_shape_mask = np.zeros_like(img_edge_mask, dtype=np.uint8)
    gelp.drawEllipse(elp_shape_mask, (255, 255, 255), thickness = 3, center_offset = [1, 1])
    
    img_edge_elp_mask = np.bitwise_and(img_edge_mask, elp_shape_mask)
    
    contours = cv2.findContours(img_edge_elp_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = contours[0]
    usage_contours = []
    for each_contour in contours:
        if len(each_contour) < 15:
            continue
        usage_contours.append(np.squeeze(np.array(each_contour))) 
        
    
    
    
    
    
    
    
    final_mask = np.zeros_like(img_edge_mask, dtype=np.uint8)
    for each_contour in usage_contours:
        # print(list(zip(each_contour[:, 0], each_contour[:, 1])))
        
        # idx_rs, idx_cs = list(zip(each_contour[:, 0], each_contour[:, 1]))
        final_mask[each_contour[:, 1], each_contour[:, 0]] = 255
        
        
    save_path = str(idx_image) + '.png'
    cv2.imwrite('./.cache/' + save_path, final_mask)
    
    
    
    # imgT = np.zeros_like(imgC, dtype=np.uint8)
    # imgT[:, :, 0] = elp_pts_mask
    # imgT[:, :, 1] = final_mask
    # imgT[:, :, 2] = elp_shape_mask
    
    
    # imgT = Image.fromarray(imgT)
    # imgT.show()
    # exit()

