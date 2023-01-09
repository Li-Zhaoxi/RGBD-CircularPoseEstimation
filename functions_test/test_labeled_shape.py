import cv2
from PoseLabel.Shapes import Shapes



img_path = '/home/expansion/lizhaoxi/datasets/BALL_INDOOR_SYNC1/images/PL-Q10F-340-.jpg'
json_path = '/home/expansion/lizhaoxi/datasets/BALL_INDOOR_SYNC1/images/PL-Q10F-340-.jpg.json'

color_ellipse = (0, 0, 255)
color_ptselp = (255, 0, 0)

color_points = (0, 255, 0)

color_lines = (0, 0, 255)
color_linepts = (255, 0, 0)


labeled_shape = Shapes()
labeled_shape.loadJson(json_path)


imgC = cv2.imread(img_path)
labeled_shape.drawAllOnImage(imgC, color_ellipse, color_ptselp, color_points, 
                             color_lines, color_linepts, thikness=1)


# cv2.imshow('imgC', imgC)
# cv2.waitKey()

cv2.imwrite('imgC.png', imgC)
