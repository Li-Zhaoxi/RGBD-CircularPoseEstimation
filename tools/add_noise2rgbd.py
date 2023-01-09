import numpy as np
import os
import cv2
from PIL import Image
from ElpPy.dataproc import GTPoseLoader
from ElpPy.dataproc import imnoise_gauss, add_gaussian_shifts, filterDisp
from tqdm import tqdm
from ElpPy.utils import depth_colorizer
import OpenEXR
import array

dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
dataset_name = 'CircularPose-15PlaneBlender'
save_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlenderNoise/'

need_color_noise = False
need_depth_noise = True

noise_mask_path = os.path.join(dataset_root_path, 'noise_masks.npz')
generate_mask = np.load(noise_mask_path, allow_pickle=True)['masks'][()]

loader = GTPoseLoader()
loader.load(dataset_root_path, dataset_name)

gt_num = len(loader)
print(gt_num)

for idx_image in tqdm(range(gt_num)):
    # print('Processing {0}th image.'.format(idx_image))
    gt = loader.__getitem__(idx_image, True)
    
    
    if need_color_noise:
        imgC = gt['imgC']
        
        color_suffix = gt['color_suffix']
        
        usage_mask = generate_mask[idx_image % 10, :, :, :]
        
        noise_imgC = imnoise_gauss(imgC, usage_mask, var=0.25)
        
        save_full_path = os.path.join(save_root_path, color_suffix)
        cv2.imwrite(save_full_path, noise_imgC)
        # show_img = Image.fromarray(noise_imgC)
        # show_img.show()
        
    if need_depth_noise:
        
        gcam = gt['camera']
        imgD = gt['imgD']
        
        # reference: https://github.com/mklingen/depth_noiser
        noise_constant = 0.001
        noise_linear = 0.01
        noise_quadratic = 0.001
        
        
        usage_mask = generate_mask[idx_image % 10, :, :, :]
        
        depth_sigma = noise_constant + imgD * noise_linear + imgD * imgD * noise_quadratic
        attach_noise = depth_sigma * usage_mask[:, :, 0]
        
        noise_depth = imgD + attach_noise
        noise_depth[noise_depth < 0] = 0
        noise_depth[imgD <= 0] = 0
        
        depth_suffix = gt['depth_suffix']
        
        save_full_path = os.path.join(save_root_path, depth_suffix)
        
        depth_rows, depth_cols = noise_depth.shape
        # final_save = np.zeros((depth_rows, depth_cols, 3))
        # final_save[:, :, 0] = noise_depth
        # final_save[:, :, 1] = noise_depth
        # final_save[:, :, 2] = noise_depth
        
        data = np.asarray(noise_depth,'f').tostring()
        # data = array.array('f', noise_depth.tolist()).tostring()
        exr = OpenEXR.OutputFile(save_full_path, OpenEXR.Header(depth_cols,depth_rows))
        exr.writePixels({'R': data, 'G': data, 'B': data})
        
        # final_save = final_save.astype('float32')
        # print(save_full_path)
        # imageio.imwrite(save_full_path, final_save)
        # cv2.imwrite(save_full_path, final_save)
        # print(np.max(attach_noise))
        
        # depth_rgb = depth_colorizer(noise_depth)
        
        # show_img = Image.fromarray(depth_rgb)
        # show_img.show()
        
        
    # exit()
        
    
