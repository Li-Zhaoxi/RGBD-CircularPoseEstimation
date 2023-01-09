import numpy as np
import random

from ElpPy.lib.pyEllipseTools import pyELPShape2Equation, pyCalculateRangeOfY, pyCalculateRangeAtY
from ElpPy.plane_fitting.plane import Plane
from ElpPy.plane_fitting.bayesplane import BayesPlane
import pyransac3d as pyrsc


def extract_ellipse_inner_points(elp, irows, icols):
    equ_mask = pyELPShape2Equation(elp)
    xmin, xmax, ymin, ymax = pyCalculateRangeOfY(elp)

    ymin = np.max([0, np.floor(ymin)])
    ymax = np.min([irows - 1, np.ceil(ymax)])

    pts_mask = []
    for idx_y in range(int(ymin), int(ymax) + 1):

        x_min_max = pyCalculateRangeAtY(equ_mask, np.float64(idx_y))
        if x_min_max is not None:
            xmin, xmax = x_min_max
            xmin = np.max([0, np.ceil(xmin)])
            xmax = np.min([icols - 1, np.floor(xmax)])
            if xmin > xmax:
                continue
            elif xmin + 1 > xmax:
                pts_mask.append([int(xmin), int(idx_y)])
            else:
                for idx_x in range(int(xmin), int(xmax) + 1):
                    pts_mask.append([int(idx_x), int(idx_y)])
    return np.stack(pts_mask)

def extract_depth_points(pts_idx, depth_img, intrinsic = None):
    pts = []

    if intrinsic is None:
        pcvt = lambda x, y, z: [x, y, z]
    else:
        fx, fy, u0, v0, factor = intrinsic
        pcvt = lambda x, y, z: [(x - u0) / fx * z * factor, (y - v0) / fy * z * factor, z * factor]
    for pt_idx in pts_idx:
        idx_col, idx_row = pt_idx
        pt = pcvt(idx_col, idx_row, depth_img[idx_row, idx_col])
        pts.append(pt)
    return np.array(pts)



def fit_plane_bayes(data, cov):
    '''
    Fits a plane and its covariance, based on a radial noise model
    @param data - points to fit plane to, (X,N) matrix
    @param cov - radial covariance of each point, (X,) vector
    @return BayesPlane class with mean and covariance
    '''
    nans = np.isnan(data[:, 0] * data[:, 1])
    data = data[~nans, :]
    cov = cov[~nans]
    if data.size == 0:
        raise ValueError('Data matrix is empty')


    # 至此Nan的数据被去除了
    w = 1 / cov
    pc = np.asarray(sum([_w * _d for _w, _d in zip(w, data)]) / sum(w)) # 点的均值
    x = data - pc
    wx = np.asarray([_w * _x for _w, _x in zip(w, x)])
    M = np.dot(wx.T, x)
    n = np.linalg.svd(M)[0][:, -1]
    d = np.dot(n.T, pc)
    plane = Plane(n, d)
    H = np.eye(plane.dim() + 1)
    H[-1, -1] = -np.sum(w)
    H[0:-1, -1] = -H[-1, -1] * pc
    H[-1, 0:-1] = -H[-1, -1] * pc
    H[0:-1, 0:-1] = -M + H[-1, -1] * np.dot(pc, pc.T) + \
        np.dot(n.T, np.dot(M, n)) * np.eye(data.shape[1])
    return BayesPlane(plane, -np.linalg.inv(H))


def fit_plane(pts3D: np.ndarray):
    # Fit a plane, a1x+a2y+a3z=d, n = [a1,a2,a3]
    
    sample_cov = 0.9
    cov = np.asarray([sample_cov] * pts3D.shape[0])
    
    # print('pts3D', pts3D)
    depth_plane_est = fit_plane_bayes(pts3D, cov)
    
    n = depth_plane_est.mean.n
    d = depth_plane_est.mean.d
    
    err = np.abs(np.matmul(np.array([n]), np.transpose(pts3D)) - d)
    
    pts_center = np.mean(pts3D, axis=0)
    if n[2] > 0:
        n = -n
        d = -d
    return n, d, err, pts_center



class Plane:
    """
    Implementation of planar RANSAC.

    Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim.

    Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.

    ![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")

    ---
    """

    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, thresh=0.05, minPoints=100, maxIteration=1000):
        """
        Find the best equation for a plane.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers

        ---
        """
        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []
        random.seed(1)
        
        usage_num = min(n_points / 4, minPoints)
        
        for it in range(maxIteration):

            # Samples 3 random points
            
            id_samples = random.sample(range(0, n_points), usage_num)
            pt_samples = pts[id_samples]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            self.inliers = best_inliers
            self.equation = best_eq

        return self.equation, self.inliers



def fit_plane_ransac_ex(pts3D: np.ndarray):
    plane1 = Plane()
    best_eq, best_inliers = plane1.fit(pts3D, 0.003, minPoints=10)
    
    n = best_eq[0:3]
    ln = np.linalg.norm(n)
    d = -best_eq[-1]
    
    n = n / ln
    
    d = d / ln
    
    if n[2] > 0:
        n = -n
        d = -d
    
    return n, d, best_inliers

def fit_plane_ransac(pts3D: np.ndarray):
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(pts3D, 0.003, minPoints=100)
    # best_eq, best_inliers = plane1.fit(pts3D, 0.01, minPoints=100)
    n = best_eq[0:3]
    ln = np.linalg.norm(n)
    d = -best_eq[-1]
    
    n = n / ln
    
    d = d / ln
    
    if n[2] > 0:
        n = -n
        d = -d
    
    return n, d, best_inliers