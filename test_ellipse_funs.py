import numpy as np
from ElpPy.utils import GeneralEllipse, ElliFit
import cv2


elp_data = np.array([100, 200, 80, 40, np.pi/3])


imgC = np.zeros((500, 500, 3), dtype='uint8') + 255
imgT = np.copy(imgC)

gelp = GeneralEllipse(elp_data, 'shape_matrix')


gelp.drawEllipse(imgC, (0, 0, 255), 2)



# print(gelp.ellipse_shape_img())



x, y = gelp.GenerateElpData(format_img=True)

pts = np.vstack([x,y]).transpose()

new_gelp = ElliFit(pts, 'shape_image')

print(gelp.ellipse_shape_img())
print(new_gelp.ellipse_shape_img())

new_gelp.drawEllipse(imgT, (0, 0, 255), 2)



cv2.imshow('imgc', imgC)
cv2.imshow('imgT', imgT)
cv2.waitKey()


