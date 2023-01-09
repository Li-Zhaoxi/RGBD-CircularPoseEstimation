import os
import cv2
from ElpPy.dataproc import GTPoseLoader
from ElpPy.utils import perspectCircular2Image
import numpy as np
from tqdm import tqdm

# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneTrain/'
dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'

dataset_name = 'CircularPose-15PlaneBlender'

loader = GTPoseLoader()

loader.load(dataset_root_path, dataset_name)

images_num = len(loader)
print(images_num)
for idx_images in tqdm(range(images_num)):
    # idx_images = 10000
    gt = loader[idx_images]
    
    gcam = gt['camera']
    gscirs = gt['spacialcircle']
    img_size = gt['size']
    
    # print(img_size)
    # exit()
    
    # 生成透视到图像上的所有椭圆
    # gelps = []
    mask = np.zeros((img_size[1], img_size[0]), dtype='uint8')
    for idx_gscir in range(len(gscirs)):
        tmpgelp = perspectCircular2Image(gcam, gscirs[idx_gscir])
        if tmpgelp.getArea() < 5 * 5 * np.pi:
            continue
        # tmpgelp.drawEllipse(mask, (255, 255, 255), -1)
        tmpgelp.drawEllipse(mask, (idx_gscir + 1, idx_gscir + 1, idx_gscir + 1), -1)
    
    # cv2.imshow('mask', mask)
    # cv2.waitKey()
    mask_path = os.path.join(dataset_root_path, 'mask', 'MASK_{0}.png'.format(idx_images))
    cv2.imwrite(mask_path, mask)
    # exit()
    
    



# img_draw = loader.drawGT(1000)

# cv2.imshow('imgT', img_draw)
# cv2.waitKey()