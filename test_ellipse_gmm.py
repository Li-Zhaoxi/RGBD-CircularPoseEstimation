import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

elp1 = np.array([100, 100, 100, 50, np.pi/3])
elp2 = np.array([100, 100, 100, 50, np.pi/4])

# 参数定义
N = 40
h = 2


# 椭圆建模

x0, y0, R, r, theta = elp1

tao = np.linspace(0, np.pi * 2, N + 1)

px = np.cos(theta) * R * np.cos(tao) - np.sin(theta) * r * np.sin(tao) + x0
py = np.sin(theta) * R * np.cos(tao) + np.cos(theta) * r * np.sin(tao) + y0
pxy = np.vstack((px, py))
ntj = np.diff(pxy)
htj = np.linalg.norm(ntj, axis=0)
h_sum = np.sum(htj)


muj = pxy[:, 0:-1] + ntj/2
muj = np.transpose(muj)
ntj = np.transpose(ntj)
wj = htj / h_sum

varj = []
for hj, n in zip(htj, ntj):
    n = n / hj
    Q = np.array([[-n[1], n[0]], [n[0], n[1]]])
    A = np.diag((h*h, hj * hj))
    varj.append(Q.dot(A).dot(Q.transpose()))
varj = np.array(varj)


# 测试建模是否正确

rows = 200
cols = 200
x = np.array(range(0, rows))
y = np.array(range(0, cols))


Y, X = np.meshgrid(y, x)

vx = X.reshape(-1, 1)
vy = Y.reshape(-1, 1)
vxy = np.hstack((vx, vy))

zxy = []
for p in vxy:
    zp = 0
    for w, mu, var in zip(wj, muj, varj):
        zp = zp + w / np.sqrt(np.linalg.det(var)) * np.exp(-0.5 * np.matmul(np.matmul(p - mu, np.linalg.inv(var)), np.transpose(p-mu)))
    zxy.append(zp)

Z = np.array(zxy).reshape(rows, cols)



sio.savemat('data.mat', {'X': X, 'Y': Y, 'Z': Z})

#print(vxy)



#print(np.diff(pxy))
#print(varj)
'''
plt.plot(px, py, ls='-', lw=2, color='purple')
ax = plt.gca()
ax.set_aspect(1)
plt.show()
'''