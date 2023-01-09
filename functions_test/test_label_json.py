from PoseLabel.Shapes import Shapes
import os
import cv2
import numpy as np

img_file_path = '/home/expansion/lizhaoxi/datasets/Pose/PlannerDatasets/light1/color/10.jpg'

json_file_path = img_file_path + '.json'

imgC = cv2.imread(img_file_path)

label_shapes = Shapes()
label_shapes.loadJson(json_file_path)

label_shapes.drawAllOnImage(imgC, (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), 
                            thikness=1)

cv2.imwrite('label.png', imgC)