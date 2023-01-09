import numpy as np
import cv2
from ElpPy.utils import depth_colorizer
from PIL import Image
depth_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/depth/DEPTH_2845.exr'
color_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/rgb/COLOR_2845.png'



imgD = cv2.imread(depth_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  

# d1 = imgD[:, :, 0]
# d2 = imgD[:, :, 1]
# d3 = imgD[:, :, 2]

# print(imgD.shape)
# print(np.max(np.abs(d1 - d2)), np.max(np.abs(d2 - d3)))


imgD = imgD[:, :, 0]

print(imgD.shape)
# exit()

imgDC = depth_colorizer(imgD)
# print(imgD)


imgC = cv2.imread(color_path)

# cv2.imshow('imgC', imgC)
# cv2.imshow('depth', imgDC)


# cv2.waitKey()

cv2.imwrite('imgC.png', imgC)
cv2.imwrite('imgDC.png', imgDC)


