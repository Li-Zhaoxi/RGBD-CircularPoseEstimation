import numpy as np
import cv2
import os
from lib.pyAAMED import pyAAMED

root_path = './simple_images'
img_idx = 3


img_path = os.path.join(root_path, 'color', str(img_idx) + '_color.png')
depth_path = os.path.join(root_path, 'depth', str(img_idx) + '_depth.png')


imgC = cv2.imread(img_path)
imgG = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

aamed = pyAAMED(721, 1281)

aamed.setParameters(3.1415926/3, 3.4,0.77)
res = aamed.run_AAMED(imgG)

pts = aamed.getEdgePoints(0)

print(pts)
aamed.drawAAMED(imgG)
#cv2.imshow("imgG", imgG)



imgT = cv2.cvtColor(imgG, cv2.COLOR_GRAY2BGR)
for pt in pts:
    cv2.circle(imgT, (pt[1], pt[0]), 2, (0,0,255))
cv2.imshow("imgG", imgT)
cv2.waitKey()


elp = res[0]
np.savez('3', elp, pts)