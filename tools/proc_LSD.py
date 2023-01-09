from ElpPy.dataproc import GTPoseLoader
import os

dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
dataset_name = 'CircularPose-15PlaneBlender'
loader = GTPoseLoader()
loader.load(dataset_root_path, dataset_name)

gt_num = len(loader)
print(gt_num)

for idx_image in range(gt_num):
    print('Processing {0}th image.'.format(idx_image))
    gt = loader[idx_image]
    
    img_path = gt['color_full_path']
    
    
    cmd_str = './LSD/bin/run_lsd {0}'.format(img_path)
    print(cmd_str)
    os.system(cmd_str)
    
    