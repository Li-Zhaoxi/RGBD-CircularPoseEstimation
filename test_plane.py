import numpy as np
import os
import cv2
from tools.plane_proc import extract_ellipse_inner_points, extract_depth_points, fit_plane_bayes
from lib import cpp_tools as cts
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle



## 半径已知的方案 ###
# 1 椭圆检测，获取目标椭圆和参与拟合所需要的像素点
# 2 根据椭圆确定ROI，平面检测算法会从其中检测出多个平面，设计算法选择椭圆目标平面，平面法向就是空间圆法相初值
# 3 利用参与拟合的像素点计算其在这个平面的投影，确定空间圆中心点
#   这时候有个分歧，直接圆拟合半径对不上，如果想利用半径信息，参考HT变换的思想，可能就是慢，需要针对这种情况编程
# 4 优化法向和平移信息，使得参与拟合的点在空间圆上的投影的误差最小。
# 5 得到的最终法向有可能是二义性的虚假解，选择最贴合法向的解作为最终的结果。

### 半径未知的方案 ###
# 1 椭圆检测，获取目标椭圆和参与拟合所需要的像素点
# 2 根据椭圆确定ROI，平面检测算法会从其中检测出多个平面，设计算法选择椭圆目标平面，平面法向就是空间圆法相初值
# 3 利用参与拟合的像素点计算其在这个平面的投影，确定空间圆中心点与半径
# 4 固定空间圆中心点位置不变，变换平面法向，获得拟合误差最小的法向
# 5 固定法向不变，重新拟合平面，根据得到的新平面，重新拟合出空间圆信息

### 需要分析的几个问题 ###
# 1 仅估计法向和中心点，不考虑利用其他信息解算出完整的位姿
# 2 空间圆的深度平面，手动标注 or 自动检测
# 3 空间圆的半径已知 or 未知。
# 4 估计法向和中心点+手动标注深度平面+已知半径 是否足够



dict = np.load('3.npz')

elp = dict['arr_0']
elp_pts = dict['arr_1']



#elp = np.array([261.2049,587.7671,156.05565,113.449425,-2.6718013])
root_path = './simple_images'
img_idx = 3

fx = 899.914
fy = 900.116
u0 = 650.407
v0 = 361.103
factor = 0.00025
R_tgt = 0.059


img_path = os.path.join(root_path, 'color', str(img_idx) + '_color.png')
depth_path = os.path.join(root_path, 'depth', str(img_idx) + '_depth.png')


imgC = cv2.imread(img_path)
imgG = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
K = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])


# 单椭圆解算位姿
#elp_pose = np.array([elp[1], elp[0], elp[3]/2, elp[2]/2,  -elp[4]/180*np.pi - np.pi/2])
elp_pose = np.array([elp[1], elp[0], elp[3]/2, elp[2]/2,  -elp[4]/180*np.pi])
equ_pose = cts.pyELPShape2Equation(elp_pose)
a1, a2, a3, a4, a5, a6 = equ_pose
#C = np.array([[a3, a2, a5], [a2, a1, a4], [a5, a4, a6]])
C = np.array([[a1, a2, a4], [a2, a3, a5], [a4, a5, a6]])
X1, X2, N1, N2 = cts.pyGetCirclePos(C, K, 0.059)


# 获取椭圆内部的点的像素坐标
elp_mask = np.array([elp[1], elp[0], elp[3]/2, elp[2]/2,  -elp[4]/180*np.pi])
pts = extract_ellipse_inner_points(elp_mask, imgG.shape[0], imgG.shape[1])


# 计算点云
instrinsic = [fx, fy, u0, v0, factor]
pts3D = extract_depth_points(pts, depth, instrinsic)

# 拟合平面, a1x+a2y+a3z=d, n = [a1,a2,a3]
sample_cov = 0.9
cov = np.asarray([sample_cov] * pts3D.shape[0])
depth_plane_est = fit_plane_bayes(pts3D, cov)
n = depth_plane_est.mean.n
d = depth_plane_est.mean.d
err = np.abs(np.matmul(np.array([n]), np.transpose(pts3D)) - d)
pts_center = np.mean(pts3D, axis=0)

# 计算在相机坐标系下，参与拟合椭圆的像素点在平面上的坐标
v = np.array(elp_pts[:,0])
u = np.array(elp_pts[:,1])
up = (u - u0) / fx
vp = (v - v0) / fy
a1, a2, a3 = n
zc = d / (a1*up + a2*vp + a3)
xc = up * zc
yc = vp * zc

# 将空间点坐标转移到平面上，尝试拟合圆
xc_mean = np.mean(xc)
yc_mean = np.mean(yc)
zc_mean = np.mean(zc)

l, m, n = n
t = np.sqrt(l * l + m * m)
R = np.array([[-m/t, -l*n/t, l], [l/t, -m*n/t, m], [0, t, n]])

xyzc = np.array([xc - xc_mean, yc - yc_mean, zc - zc_mean])
xyzcp = np.matmul(R.transpose(), xyzc)
xcp = xyzcp[0]
ycp = xyzcp[1]



# 使用已知半径，精炼位置
xycp = np.array([xcp, ycp])
xycpt = xycp.transpose()
fircircle = cts.pyFitCircle(xycp)

xycp_normalize  = xycpt - fircircle[:2]

p_centers = []
T_pt_dist = R_tgt * np.sqrt(2 - 2 * np.cos(np.pi/4))
for idxi, pti in enumerate(xycp_normalize[:-1]):
    for idxj, ptj in enumerate(xycp_normalize[idxi+1:]):
        idxj = idxi + idxj + 1
        nij = ptj - pti
        lnij = np.linalg.norm(nij)
        nij = nij / lnij
        if lnij < T_pt_dist:
            continue
        p_mid_ij = (pti + ptj) / 2
        cos_ij = np.dot(pti, ptj) / (np.linalg.norm(pti) * np.linalg.norm(ptj))
        dist_tgt = R_tgt * np.sqrt((cos_ij + 1)/2)

        if abs(dist_tgt) < 1e-3:
            continue
        dir = -np.sign(np.cross(ptj, pti)) * np.array([-nij[1], nij[0]])
        pc = p_mid_ij + dir * dist_tgt
        #print(pti, ptj, dist_tgt, p_mid_ij, pc)
        #exit()
        p_centers.append(pc)
p_centers = np.array(p_centers)

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(p_centers, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(p_centers)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result

print(cluster_centers)
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(p_centers[my_members, 0], p_centers[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

exit()

print(p_centers.shape)
#plt.plot(xcp - fircircle[0], ycp - fircircle[1], lw=2, label="plot figure")
plt.plot(p_centers[:, 0], p_centers[:, 1], '.r', label="plot figure")
plt.legend()
plt.show()


plt.plot(xcp - fircircle[0], ycp - fircircle[1], lw=2, label="plot figure")
plt.legend()
plt.show()
print(xycp_theta/np.pi*180)
#print(xycp_theta/np.pi*180)
exit()
#P1 = xycp_torch.repeat(xycp.shape[0], 1, 1)
#P2 = P1.transpose(0, 1)




err = np.abs(np.sqrt(np.square(xcp - fircircle[0]) + np.square(ycp - fircircle[1])) - fircircle[2]/2) * 100

print(np.mean(err))
print(np.max(err))
print(np.min(err))

Co = np.array([[fircircle[0], fircircle[1], 0]])
CC = np.matmul(R, Co.transpose()).transpose() + np.array([xc_mean, yc_mean, zc_mean])
#
print(np.linalg.norm(CC - X1) * 1000, np.linalg.norm(CC - X2) * 1000)


'''
for xy in xycp:
    print(xy[0], xy[1])
'''
'''
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xcp, ycp, xyzcp[2], s=1)

plt.show()
'''

# 剩余的一些工作：
# 1 (完成) 单椭圆解算位姿，用于测量法向误差
# 2 (完成) 获取参与拟合椭圆的像素点。主要是获取弧段与edgeContours之间的关系，C++版本已经测试完成，需要想办法套上Python的外壳
# 3 (完成) 计算在平面上的投影点，拟合圆，计算中心点
# 4 (4.30) 构建cost函数
# 5 (5.1) 优化算法，获得cost最小的位姿信息