import numpy as np
from tools.plane_proc import extract_ellipse_inner_points, extract_depth_points, fit_plane_bayes
from lib import cpp_tools as cts
import cv2
from scipy.optimize import least_squares, fmin_l_bfgs_b, leastsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import circle_fit as cirf
from posecore.core import get_norm_Rotation, norm_dist, draw_circle_pose
from scipy.io import savemat


class PerspectiveCircleDepth(object):
    def __init__(self, knownR, intrinsic):
        self.knownR = knownR
        Kshape = np.array(intrinsic).shape

        if len(Kshape) == 1:
            fx, fy, u0, v0, factor = intrinsic
            self.fx = fx
            self.fy = fy
            self.u0 = u0
            self.v0 = v0
            self.factor = factor
            self.intrinsic = [fx, fy, u0, v0, factor]
        else:
            raise Exception("Invalid intrinsic!", intrinsic)

        self.K = np.array([[self.fx, 0, self.u0], [0, self.fy, self.v0], [0, 0, 1]])

    @classmethod
    def fit_plane(cls, pts3D):
        # Fit a plane, a1x+a2y+a3z=d, n = [a1,a2,a3]
        sample_cov = 0.9
        cov = np.asarray([sample_cov] * pts3D.shape[0])
        depth_plane_est = fit_plane_bayes(pts3D, cov)
        n = depth_plane_est.mean.n
        d = depth_plane_est.mean.d
        err = np.abs(np.matmul(np.array([n]), np.transpose(pts3D)) - d)
        pts_center = np.mean(pts3D, axis=0)
        if n[2] > 0:
            n = -n
            d = -d
        return n, d, err, pts_center

    @classmethod
    def get_error_Rotation(cls, eta, xi):
        cos_xi = np.cos(xi)
        sin_xi = np.sin(xi)
        cos_eta = np.cos(eta)
        sin_eta = np.sin(eta)

        Re = np.array([[-sin_eta, -cos_eta * cos_xi, cos_eta * sin_xi],
                       [cos_eta, -sin_eta * cos_xi, sin_eta * sin_xi],
                       [0, sin_xi, cos_xi]])
        return Re

    def perspective_pixels2circle(self, pts, circle_norm, circle_pos):
        circle_plane_d = np.dot(circle_norm, circle_pos)
        v = np.array(pts[:, 0])
        u = np.array(pts[:, 1])
        up = (u - self.u0) / self.fx
        vp = (v - self.v0) / self.fy
        a1, a2, a3 = circle_norm
        zc = circle_plane_d / (a1 * up + a2 * vp + a3)
        xc = np.multiply(up, zc)
        yc = np.multiply(vp, zc)

        xc_mean, yc_mean, zc_mean = circle_pos

        xyzc = np.array([xc - xc_mean, yc - yc_mean, zc - zc_mean])

        R = get_norm_Rotation(circle_norm)

        xyzcp = np.matmul(R.transpose(), xyzc)
        px = xyzcp[0]
        py = xyzcp[1]

        return px, py, R, np.array([xc_mean, yc_mean, zc_mean])

    def perspective_pixels2plane(self, pts, n, d):
        v = np.array(pts[:, 0])
        u = np.array(pts[:, 1])
        up = (u - self.u0) / self.fx
        vp = (v - self.v0) / self.fy
        a1, a2, a3 = n
        zc = d / (a1 * up + a2 * vp + a3)
        xc = up * zc
        yc = vp * zc

        xc_mean = np.mean(xc)
        yc_mean = np.mean(yc)
        zc_mean = np.mean(zc)
        xyzc = np.array([xc - xc_mean, yc - yc_mean, zc - zc_mean])

        R = get_norm_Rotation(n)

        xyzcp = np.matmul(R.transpose(), xyzc)

        px = xyzcp[0]
        py = xyzcp[1]

        return px, py, R, np.array([xc_mean, yc_mean, zc_mean])

    def calculateSingleCirclePose(self, elps, isBatch=True):
        if isBatch:
            poses = []
            for elp in elps:
                poses.append(self.__GetSingleCirclePose(elp))
            return poses
        else:
            return self.__GetSingleCirclePose(elps)

    def __GetSingleCirclePose(self, elp):

        shape = elp.shape
        if len(shape) == 1:
            if shape[0] == 5:
                equ_pose = cts.pyELPShape2Equation(elp)
            else:
                equ_pose = elp
            a1, a2, a3, a4, a5, a6 = equ_pose
            # C = np.array([[a3, a2, a5], [a2, a1, a4], [a5, a4, a6]])
            C = np.array([[a1, a2, a4], [a2, a3, a5], [a4, a5, a6]])
        else:
            C = elp
        X1, X2, N1, N2 = cts.pyGetCirclePos(C, self.K, self.knownR)
        return X1, X2, N1, N2

    @classmethod
    def extract_mast_points(cls, mask):
        return np.argwhere(mask > 0)

    @classmethod
    def ellipse_inner_points(cls, elp, shape):
        return extract_ellipse_inner_points(elp, shape[0], shape[1])

    def transform_Pixel2CameraFrame(self, pixs, depth):
        pts3D = extract_depth_points(pixs, depth, self.intrinsic)
        return pts3D

    def fit_circle(self, px, py):
        xycp = np.array([px, py])
        xo, yo, R, var = cirf.hyper_fit(xycp.transpose())
        cir_parms = np.array([xo, yo, R])

        if self.knownR is not None:
            pass

        return cir_parms[0:3]

        def residual(x):
            xo = x[0]
            yo = x[1]
            square_r = np.square(px - xo) + np.square(py - yo)
            err_abs = np.abs(np.sqrt(square_r) - self.knownR)
            cost = np.sum(err_abs)
            # print(x, cost, np.sqrt(np.mean(square_r)))
            return cost

        cen = cir_parms[0:2]
        bounds = ([cen[0] - self.knownR, cen[1] - self.knownR], [cen[0] + self.knownR, cen[1] + self.knownR])

        # res = fmin_l_bfgs_b(residual, x0=cen, approx_grad=True)
        res = least_squares(residual, cen, verbose=0)

        xo = res.x[0]
        yo = res.x[1]
        res_r2 = np.square(px - xo) + np.square(py - yo)
        res_r2 = np.mean(res_r2)
        res_r = np.sqrt(res_r2)

        cir_parms_final = np.array([res.x[0], res.x[1], res_r])
        # print(cir_parms_final, self.knownR)
        return cir_parms_final

    def cost_perspective_pixels2plane(self, pts_imageframe, circle_norm, circle_pose, cost_type='Rtr'):
        px, py, R, t = self.perspective_pixels2circle(pts_imageframe, circle_norm, circle_pose)

        if cost_type == 'Rtr':
            err_abs = np.abs(np.sqrt(np.square(px) + np.square(py)) - self.knownR)
            return np.sum(err_abs)
        elif cost_type == 'R':
            xycp = np.array([px, py])
            xo, yo, R, var = cirf.hyper_fit(xycp.transpose())

            # savemat(str(circle_norm) + 'pxy.mat', {'px': px, 'py': py, 'xo': xo, 'yo': yo, 'r': R})

            dx = px - xo
            dy = py - yo
            err_abs = np.abs(np.sqrt(np.square(dx) + np.square(dy)) - R)
            return np.sum(err_abs), np.var(err_abs), R

        else:
            pass

    @classmethod
    def corrected_norm(cls, init_norm, eta, xi):
        R = get_norm_Rotation(init_norm)
        Re = cls.get_error_Rotation(eta, xi)
        circle_norm = np.matmul(R, Re)[:, 2]
        return circle_norm

    def corrected_pose(self, pts_elprim_pixelframe, final_norm, init_pos):
        px, py, R, t = self.perspective_pixels2circle(pts_elprim_pixelframe, final_norm, init_pos)
        xycp = np.array([px, py])
        xo, yo, cR, _ = cirf.hyper_fit(xycp.transpose())

        # print(xo, yo, R)
        # print(np.matmul(R, np.array([[xo, yo, 0]]).transpose()))

        correct_pos = np.matmul(R, np.array([[xo, yo, 0]]).transpose()).transpose()[0] + init_pos

        return correct_pos, cR

    def perspective_circle2image(self, circle_norm, circle_pos):
        Rwc = get_norm_Rotation(circle_norm)

        circle_pos = np.array([circle_pos])
        xyzo = np.matmul(Rwc.transpose(), -circle_pos.transpose()).transpose()
        xo, yo, zo = xyzo[0]

        U = np.array([[zo * zo, 0, -xo * zo],
                      [0, zo * zo, -yo * zo],
                      [-xo * zo, -yo * zo, xo * xo + yo * yo - self.knownR * self.knownR]])
        RwcK_inv = np.matmul(Rwc.transpose(), np.linalg.inv(self.K))
        C = np.matmul(np.matmul(RwcK_inv.transpose(), U), RwcK_inv)
        # print(xo, yo, zo)
        # print(C)
        X1, X2, N1, N2 = self.__GetSingleCirclePose(C)

        return X1, X2, N1, N2

    def best_pose_selection(self, circle_norm, circle_pos, pt_cameraframe):
        X1, X2, N1, N2 = self.perspective_circle2image(circle_norm, circle_pos)

        circle_plane_d1 = np.dot(N1, X1)
        circle_plane_d2 = np.dot(N2, X2)

        pt_err1 = np.abs(np.matmul(N1, pt_cameraframe.transpose()) - circle_plane_d1)
        pt_err2 = np.abs(np.matmul(N2, pt_cameraframe.transpose()) - circle_plane_d2)

        sum_err1 = np.sum(pt_err1)
        sum_err2 = np.sum(pt_err2)

        if sum_err1 > sum_err2:
            return N2, X2
        else:
            return N1, X1

    # pts3D: N * 3
    def findBestPlane(self, pts3D, init_norm, init_pos):

        init_norm = init_norm / np.linalg.norm(init_norm)
        init_d = np.dot(init_pos, init_norm)

        init_norm31 = np.array([init_norm]).transpose()

        d_var = 0.05

        def residual(d):
            pts_dist = (np.matmul(pts3D, init_norm31) - d) / d_var  # N * 1
            pts_loss = 1 - np.mean(np.exp(-0.5 * np.square(pts_dist)))
            print(pts_loss)
            # print(pts_loss, np.sum(np.abs(pts_dist)), np.mean(np.abs(pts_dist)))
            return pts_loss

        res = least_squares(residual, init_d, verbose=0, max_nfev=10000)
        # print(init_d, res.x[0])
        # return init_d
        return res.x[0]

    def recoverPoseRadiusfromBestPlane(self, pts_elprim_pixelframe, plane_norm, plane_d):
        px, py, R, t = self.perspective_pixels2plane(pts_elprim_pixelframe, plane_norm, plane_d)

        # corrected_pos = self.corrected_pose(pts_elprim_pixelframe, plane_norm, t)
        # return corrected_pos

        xycp = np.array([px, py])
        xo, yo, circle_R, _ = cirf.hyper_fit(xycp.transpose())

        correct_pos = np.matmul(R, np.array([[xo, yo, 0]]).transpose()).transpose()[0] + t
        return correct_pos, circle_R

    def initial_estimation(self, p_eI, p_cCSP):
        # Step 1: Plane Fitting, obtain circle norm
        init_circle_norm, d, err, pts_center = self.fit_plane(p_cCSP)

        # print(init_circle_norm)
        # exit()

        # Step 2: Fit space circle, obtain circle pos
        px, py, R, t = self.perspective_pixels2plane(p_eI, init_circle_norm, d)
        cir_parms = self.fit_circle(px, py)
        cir_pos = np.array([[cir_parms[0], cir_parms[1], 0]]).transpose()
        init_circle_pos = (np.matmul(R, cir_pos).transpose() + t)[0]
        init_R = cir_parms[2]

        return init_circle_norm, init_circle_pos, init_R

    @classmethod
    def refine_pos_radius(cls, pc_csp, nc, tc, rc, delta=0.01):
        nc = nc / np.linalg.norm(nc)
        ncM = np.array([nc]).transpose()  # 3*1
        tmp_nctxcsp = np.matmul(pc_csp, ncM)  # N * 1
        tmp_nctxtc = np.dot(nc, tc)

        sc = 1

        def residual(sc):
            err = sc * tmp_nctxtc - tmp_nctxcsp
            cost = err.transpose()[0] / delta  # constraint delta = 1
            return cost

        res = least_squares(residual, sc, verbose=0, max_nfev=100, loss='huber')
        sc = res.x

        final_tc = sc * tc
        final_rc = sc * rc
        return final_tc, final_rc, sc

    '''
    p_eI: 椭圆像素
    p_cCSP: 空间圆平面3D点，N*3
    init_n, init_t: 直接求解得到的位姿
    delta：Huber需要的参数
    '''

    def refinement(self, p_eI, p_cCSP, init_n, init_t, delta=0.01):

        # Start Orierntation Refinement
        init_n = init_n / np.linalg.norm(init_n)
        x0 = init_n

        def residual(x):
            circle_norm = np.array(x)
            circle_norm = circle_norm / np.linalg.norm(circle_norm)
            circle_pos = init_t
            cost, cost_var, r = self.cost_perspective_pixels2plane(p_eI, circle_norm, circle_pos, cost_type='R')

            cost += np.abs(norm_dist(circle_norm, init_n)) / 180.0 * np.pi
            # print(cost, cost_var, r, cost / r)
            # print(cost, circle_norm, r)
            # print(cost)
            return cost

        res = least_squares(residual, x0, verbose=0, max_nfev=10000)

        refined_orierntation = np.array(res.x)
        final_orierntation = refined_orierntation / np.linalg.norm(refined_orierntation)
        corrected_pos, corrected_radius = self.corrected_pose(p_eI, final_orierntation, init_t)

        final_tc, final_rc, sc = self.refine_pos_radius(p_cCSP, final_orierntation, corrected_pos, corrected_radius,
                                                        delta)

        return final_orierntation, final_tc, final_rc, sc

    def drawPCD(self, imgC, poses, pts, thickness=1):
        imgI = np.copy(imgC)
        init_pose = poses['initial']
        draw_circle_pose(imgI, self.K, init_pose[0], init_pose[1], init_pose[2], elp_pts=pts, thickness=thickness)

        imgOR = np.copy(imgC)
        orefine_pose = poses['medium']
        draw_circle_pose(imgOR, self.K, orefine_pose[0], orefine_pose[1], orefine_pose[2], elp_pts=pts,
                         thickness=thickness)

        imgR = np.copy(imgC)
        final_pose = poses['final']
        draw_circle_pose(imgR, self.K, final_pose[0], final_pose[1], final_pose[2], elp_pts=pts, thickness=thickness)

        return imgI, imgOR, imgR

    # 在Safaee的基础上，单纯利用深度图去除法向
    def perspectCircleDepthSimple(self, depth, pts_circleplane_pixelframe, det_elp_shape):
        # convert plane pixels to the camera frame
        pts_cameraframe = self.transform_Pixel2CameraFrame(pts_circleplane_pixelframe, depth)
        init_circle_norm, d, err, pts_center = self.fit_plane(pts_cameraframe)

        # 解算两个椭圆的pose
        equ_pos = cts.pyELPShape2Equation(det_elp_shape)
        a1, a2, a3, a4, a5, a6 = equ_pos
        C = np.array([[a1, a2, a4], [a2, a3, a5], [a4, a5, a6]])
        X1, X2, N1, N2 = cts.pyGetCirclePos(C, self.K, self.knownR)

        dist1 = norm_dist(N1, init_circle_norm)
        dist2 = norm_dist(N2, init_circle_norm)

        if dist1 > dist2:
            return {'final': (N2, X2, self.knownR)}
        else:
            return {'final': (N1, X1, self.knownR)}

    # 直线与某个轴平行，因为只考虑法向估计的精度，因此可以简化问题
    @classmethod
    def recoverFullPoseCase1(cls, n_pil: np.ndarray, n_circleO: np.ndarray, n_circle: np.ndarray, pos: np.ndarray):
        sign_n_pil = np.dot(n_pil, n_circleO)

        n_pi1 = n_pil if sign_n_pil >= 0 else -n_pil
        n_ZB = n_circle if n_circle[2] >= 0 else -n_circle

        n_YB = np.cross(n_ZB, n_pi1)

        # print('n_YB, n_ZB', n_YB, n_ZB)
        n_XB = np.cross(n_YB, n_ZB)

        n_XB /= np.linalg.norm(n_XB)
        n_YB /= np.linalg.norm(n_YB)
        n_ZB /= np.linalg.norm(n_ZB)

        R = np.vstack((n_XB, n_YB, n_ZB)).transpose()

        T = np.zeros((4, 4), dtype=np.float)

        T[0:3, 0:3] = R
        T[0:3, 3] = pos
        T[3, 3] = 1.0

        return T

    @classmethod
    def fitLinefrom2Pts(cls, pst: np.ndarray, ped: np.ndarray):
        dx = ped[0] - pst[0]
        dy = ped[1] - pst[1]
        x0 = pst[0]
        y0 = pst[1]

        fitted_line = np.array([dx, dy, x0, y0])
        line_p = fitted_line[1]
        line_q = -fitted_line[0]
        line_k = -fitted_line[1] * fitted_line[2] + fitted_line[0] * fitted_line[3]

        return line_p, line_q, line_k

    def perspective_circle_line(self, line_pts: np.ndarray, det_elp_shape, line_pt_B: np.ndarray):
        # print(self.K)
        # print(line_pt_B)
        # 解算两个椭圆的pose
        equ_pos = cts.pyELPShape2Equation(det_elp_shape)
        a1, a2, a3, a4, a5, a6 = equ_pos
        C = np.array([[a1, a2, a4], [a2, a3, a5], [a4, a5, a6]])
        X1, X2, N1, N2 = cts.pyGetCirclePos(C, self.K, self.knownR)

        # 拟合直线,前两个元素为方向，后两个为直线上的一个点
        fitted_line = np.array(cv2.fitLine(line_pts, cv2.DIST_L2, 0, 1e-2, 1e-2)).transpose()[0]
        line_p = fitted_line[1]
        line_q = -fitted_line[0]
        line_k = -fitted_line[1] * fitted_line[2] + fitted_line[0] * fitted_line[3]

        # print(fitted_line, [line_p, line_q, line_k])
        # print(np.matmul(np.array([[line_p, line_q, line_k]]), self.K))

        n_pi1 = np.matmul(np.array([[line_p, line_q, line_k]]), self.K)[0]
        n_pi1 /= np.linalg.norm(n_pi1)
        n_pi1 = n_pi1 if n_pi1[0] >= 0 else -n_pi1  # 直线与相机坐标系原点构建的平面方程的法向
        # print('n_pi1', n_pi1)

        # 恢复两个位姿解的完整pose
        u_cirO = det_elp_shape[0]
        v_cir0 = det_elp_shape[1]
        n_cirO = np.matmul(np.linalg.inv(self.K), np.array([[u_cirO], [v_cir0], [1]]))
        n_cirO = n_cirO.transpose()[0]
        n_cirO /= np.linalg.norm(n_cirO)
        # print(n_cirO, det_elp_shape)
        # exit()
        T1 = self.recoverFullPoseCase1(n_pi1, n_cirO, N1, X1)
        T2 = self.recoverFullPoseCase1(n_pi1, n_cirO, N2, X2)
        # print(N1, X1)
        # print('T1', T1)
        # print(N2, X2)
        # print('T2', T2)



        # 确定合作直线在图像上的理想投影line_pt_B
        pt_B_st = line_pt_B[0]
        pt_B_ed = line_pt_B[1]
        line_Bpts = np.array([[pt_B_st[0], pt_B_st[1], pt_B_st[2], 1], [pt_B_ed[0], pt_B_ed[1], pt_B_ed[2], 1]]).transpose()
        line_Ipts1 = np.matmul(self.K, np.matmul(T1, line_Bpts)[0:3, :])
        line_Ipts1 = line_Ipts1[:-1, :] / line_Ipts1[-1, :].transpose()
        line_Ipts1 = line_Ipts1.transpose()
        # print('line_Ipts1', line_Ipts1)

        line_Ipts2 = np.matmul(self.K, np.matmul(T2, line_Bpts)[0:3, :])
        line_Ipts2 = line_Ipts2[:-1, :] / line_Ipts2[-1, :].transpose()
        line_Ipts2 = line_Ipts2.transpose()
        # print('line_Ipts2', line_Ipts2)

        line_p1, line_q1, line_k1 = self.fitLinefrom2Pts(line_Ipts1[0, :], line_Ipts1[1, :])
        line_p2, line_q2, line_k2 = self.fitLinefrom2Pts(line_Ipts2[0, :], line_Ipts2[1, :])

        # 计算直线上的点到这两个直线段的距离误差
        pts_num = line_pts.shape[0]
        line_err1 = np.zeros(pts_num, dtype=np.float)
        line_err2 = np.zeros(pts_num, dtype=np.float)
        for idx_pts in range(pts_num):
            usage_pt = line_pts[idx_pts]
            u, v = usage_pt
            dst1 = abs(u * line_p1 + v * line_q1 + line_k1) \
                   / np.sqrt(line_p1 * line_p1 + line_q1 * line_q1)
            dst2 = abs(u * line_p2 + v * line_q2 + line_k2) \
                   / np.sqrt(line_p2 * line_p2 + line_q2 * line_q2)
            line_err1[idx_pts] = dst1
            line_err2[idx_pts] = dst2

        avg_err1 = np.mean(line_err1)
        avg_err2 = np.mean(line_err2)

        # 选择最小误差对应的法向和位姿作为最终输出



        # print('avg_err1', avg_err1)
        # print('avg_err2', avg_err2)

        if avg_err1 > avg_err2:
            return {'final': (N2, X2, self.knownR), 'line_Ipts': line_Ipts2}
        else:
            return {'final': (N1, X1, self.knownR), 'line_Ipts': line_Ipts1}

        exit()
        n_p1 = np.matmul(np.array([[-fitted_line[1], fitted_line[0],
                                    fitted_line[1] * fitted_line[3] - fitted_line[0] * fitted_line[2]]]), self.K)
        # print('n_p1', n_p1)
        n_p1 = n_p1[0]
        n_p1 = n_p1 / np.linalg.norm(n_p1)

        T1 = self.recoverFullPoseCase1(n_p1, N1, X1)
        print(T1)
        # print(np.array([[-fitted_line[1], fitted_line[0],
        #                     fitted_line[1] * fitted_line[3] - fitted_line[0] * fitted_line[2]]]))
        print(n_p1)
        print(N1)
        # NY1 = np.cross(fitted_line[0:2], N1)
        # print(NY1)

        print(fitted_line)
        # 恢复完整pose

        # 去除二义性

    # pts_circleplane_pixelframe：第一列为列标，第二列为行标
    # pts_elprim_pixelframe： 第一列为行标，第二列为列标
    def __call__(self, depth, pts_circleplane_pixelframe, pts_elprim_pixelframe):

        # convert plane pixels to the camera frame
        pts_cameraframe = self.transform_Pixel2CameraFrame(pts_circleplane_pixelframe, depth)

        # Initial Estimation
        init_orierntation, init_position, init_radius = self.initial_estimation(pts_elprim_pixelframe, pts_cameraframe)

        # Refinement
        final_orierntation, final_position, final_radius, sc = self.refinement(pts_elprim_pixelframe, pts_cameraframe,
                                                                               init_orierntation, init_position)

        return {'initial': (init_orierntation, init_position, init_radius),
                'medium': (final_orierntation, final_position / sc, final_radius / sc),
                'final': (final_orierntation, final_position, final_radius)}

        # Step 1: Plane Fitting, obtain circle norm

        init_circle_norm, d, err, pts_center = self.fit_plane(pts_cameraframe)

        # Step 2: Fit space circle, obtain circle pos
        px, py, R, t = self.perspective_pixels2plane(pts_elprim_pixelframe, init_circle_norm, d)
        cir_parms = self.fit_circle(px, py)
        cir_pos = np.array([[cir_parms[0], cir_parms[1], 0]]).transpose()
        init_circle_pos = (np.matmul(R, cir_pos).transpose() + t)[0]

        savemat('data.mat', {'pts_cameraframe': pts_cameraframe, 'init_circle_norm': init_circle_norm, 'd': d, 'px': px,
                             'py': py, 'R': R, 't': t, 'init_circle_pos': init_circle_pos,
                             'pts_elprim_pixelframe': pts_elprim_pixelframe,
                             'cir_parms': cir_parms, 'K': self.K})
        exit()

        log_x = []

        # init_circle_norm = np.array([-0.87862376,  0.00538839, -0.47748429])
        # init_circle_norm = np.array([-8.66025448e-01, 1.46208405e-08, -4.99999940e-01])
        x0 = init_circle_norm

        def residual(x):
            circle_norm = np.array(x)
            circle_norm = circle_norm / np.linalg.norm(circle_norm)
            circle_pos = init_circle_pos
            cost, cost_var, r = self.cost_perspective_pixels2plane(pts_elprim_pixelframe, circle_norm, circle_pos,
                                                                   cost_type='R')
            # print(cost, cost_var, r, cost / r)
            # print(cost)
            return cost

        res = least_squares(residual, x0, verbose=0, max_nfev=10000)
        corrected_norm = np.array(res.x)
        corrected_norm = corrected_norm / np.linalg.norm(corrected_norm)
        corrected_pos = self.corrected_pose(pts_elprim_pixelframe, corrected_norm, init_circle_pos)

        d_correct = np.dot(corrected_pos, corrected_norm)
        d_best = self.findBestPlane(pts_cameraframe, corrected_norm, corrected_pos)
        print('d_best', d_best, 'd_correct', d_correct)

        # # print(corrected_pos, np.dot(corrected_pos, corrected_norm), d_best)
        # corrected_pos, circle_r = self.recoverPoseRadiusfromBestPlane(pts_elprim_pixelframe, corrected_norm, d_best)
        # # print(corrected_pos)
        # print(circle_r)

        # 下一步就是拟合出位置信息，构建每个点的高斯混合模型，再构建平面的高斯模型
        # ① 先补偿初始位姿
        # ②

        # px, py, R, t = self.perspective_pixels2circle(pts_elprim_pixelframe, corrected_norm, corrected_pos)
        # xycp = np.array([px, py])
        #
        #
        # xo, yo, r, var = cirf.hyper_fit(xycp.transpose())
        #
        # dx = px - xo
        # dy = py - yo
        # err_abs = np.abs(np.sqrt(np.square(dx) + np.square(dy)) - r)
        # print('err_abs', np.sum(err_abs))
        #
        # savemat('res.mat', {'px': px, 'py': py, 'xo': xo, 'yo': yo, 'r': r})
        # exit()

        # Step 5: Final Pose Selection
        # print(res.x)
        final_norm, final_pos = self.best_pose_selection(corrected_norm, corrected_pos, pts_cameraframe)
        return {'init_pose': (init_circle_norm, init_circle_pos),
                'refine_pose': (corrected_norm, corrected_pos),
                'final_pose': (final_norm, final_pos),
                'log_refine': log_x}
