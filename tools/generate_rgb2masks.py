import os
import numpy as np
from PIL import Image
import cv2
from PoseLabel.Shapes import Shapes

# ds_root_path = '/home/expansion/lizhaoxi/datasets/Pose/PlannerDatasets/light1/'
ds_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark3/'


usage_root = ds_root_path
usage_json_names = []
usage_image_names = []
for root, dirs, files in os.walk(os.path.join(ds_root_path, 'gt')):
    files = sorted(files)
    usage_json_names = files
    
    
for idx_name in range(len(usage_json_names)):
    print(usage_json_names[idx_name].strip('.json'))
    # continue
    img_tmp = os.path.join(usage_root, 'color', usage_json_names[idx_name].strip('.json'))
    json_file_path = os.path.join(usage_root, 'gt', usage_json_names[idx_name])
    label_shapes = Shapes()
    label_shapes.loadJson(json_file_path)
    imgC = cv2.imread(img_tmp)
    mask = np.zeros((imgC.shape[0], imgC.shape[1]), dtype='uint8')
    
    elps = label_shapes.labeled_elps

    elps = [t['gelp'] for t in elps]
    
    # print(len(elps), json_file_path)

    area1 = elps[0].getArea()
    area2 = elps[1].getArea()

    if area1 > area2:
        elps[0].drawEllipse(mask, (1, 1, 1), thickness = -1)
        elps[1].drawEllipse(mask, (0, 0, 0), thickness = -1)
    else:
        elps[1].drawEllipse(mask, (1, 1, 1), thickness = -1)
        elps[0].drawEllipse(mask, (0, 0, 0), thickness = -1)
        
    mask_path = os.path.join(usage_root, 'mask', usage_json_names[idx_name].strip('.json') + '.png')
    # print(mask_path)
    cv2.imwrite(mask_path, mask)
    
    







# label_shapes = Shapes()
# label_shapes.loadJson(json_file_path)
# imgC = cv2.imread(img_tmp)
# label_shapes.drawAllOnImage(imgC, (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), 
#                             thikness=1)



# mask = np.zeros((imgC.shape[0], imgC.shape[1]), dtype='uint8')
# elps = label_shapes.labeled_elps

# elps = [t['gelp'] for t in elps]

# area1 = elps[0].getArea()
# area2 = elps[1].getArea()

# if area1 > area2:
#     elps[0].drawEllipse(mask, (255, 255, 255), thickness = -1)
#     elps[1].drawEllipse(mask, (0, 0, 0), thickness = -1)
# else:
#     elps[1].drawEllipse(mask, (255, 255, 255), thickness = -1)
#     elps[0].drawEllipse(mask, (0, 0, 0), thickness = -1)

# # print(img_tmp)





# image = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
# image.show()
