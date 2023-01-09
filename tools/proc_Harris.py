from ElpPy.dataproc import GTPoseLoader
import os
import cv2 
import numpy as np
from PIL import Image
import json


dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
dataset_name = 'CircularPose-15PlaneBlender'
loader = GTPoseLoader()
loader.load(dataset_root_path, dataset_name)

gt_num = len(loader)
print(gt_num)

for idx_image in range(gt_num):
    print('Processing {0}th image.'.format(idx_image))
    gt = loader.__getitem__(idx_image, True)
    
    imgG = gt['imgG']
    imgC = gt['imgC']
    img_path = gt['color_full_path']
    
    imgG = np.float32(imgG)
    
    dst = cv2.cornerHarris(imgG, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    
    pi, pj = np.where(dst > 0.005*dst.max())
    
    save_dict = {}
    save_dict['pi'] = pi.tolist()
    save_dict['pj'] = pj.tolist()
    
    save_path = img_path + '.json'
    
    json_str = json.dumps(save_dict, indent=1)
    
    with open(save_path, 'w') as f:
        f.write(json_str)
        f.close()
    
    # imgC[pi, pj, :] = [0, 0, 255]
    # # imgC[dst > 0.005*dst.max()] = [255, 0, 0]
    
    # image = Image.fromarray(imgC)
    # image.show()
 
    # exit()
    
    
    