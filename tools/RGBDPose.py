import numpy as np
from tools.plane_proc import extract_ellipse_inner_points, extract_depth_points, fit_plane_bayes
from lib import cpp_tools as cts
import cv2
from scipy.optimize import least_squares, fmin_l_bfgs_b, leastsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import circle_fit as cirf


class PerspectiveCircleDepth(object):
    def __init__(self, knownR, intrinsic):
        self.knownR = knownR
        fx, fy, u0, v0, factor = intrinsic
        self.fx = fx
        self.fy = fy
        self.u0 = u0
        self.v0 = v0
        self.factor = factor
        self.intrinsic = intrinsic
        self.K = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])

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
    def get_Rotation(cls, n):
        l, m, n = n
        t = np.sqrt(l * l + m * m)
        if np.abs(t) < 1e-6:
            return np.eye(3)
        R = np.array([[-m / t, -l * n / t, l], [l / t, -m * n / t, m], [0, t, n]])
        return R

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

        R = self.get_Rotation(circle_norm)

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


        R = self.get_Rotation(n)

        xyzcp = np.matmul(R.transpose(), xyzc)




        px = xyzcp[0]
        py = xyzcp[1]

        return px, py, R, np.array([xc_mean, yc_mean, zc_mean])

    def GetSingleCirclePose(self, elp):

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
    def ellipse_inner_points(cls, elp, shape):
        return extract_ellipse_inner_points(elp, shape[0], shape[1])

    def transform_Pixel2CameraFrame(self, pixs, depth):
        pts3D = extract_depth_points(pixs, depth, self.intrinsic)
        return pts3D


    def fit_circle(self, px, py):
        xycp = np.array([px, py])


        #print(py)

        #cir_parms = cts.pyFitCircle(xycp)
        xo, yo, R, var = cirf.hyper_fit(xycp.transpose())
        cir_parms = np.array([xo,yo,R])

        #print(cir_parms)

        return cir_parms[0:3]

        def residual(x):
            xo = x[0]
            yo = x[1]
            square_r = np.square(px - xo) + np.square(py - yo)
            err_abs = np.abs(np.sqrt(square_r) - self.knownR)
            cost = np.sum(err_abs)
            #print(x, cost, np.sqrt(np.mean(square_r)))
            return cost

        cen = cir_parms[0:2]
        bounds = ([cen[0]-self.knownR, cen[1]-self.knownR], [cen[0]+self.knownR, cen[1]+self.knownR])

        #res = fmin_l_bfgs_b(residual, x0=cen, approx_grad=True)
        res = least_squares(residual, cen, verbose=0)

        xo = res.x[0]
        yo = res.x[1]
        res_r2 = np.square(px - xo) + np.square(py - yo)
        res_r2 = np.mean(res_r2)
        res_r = np.sqrt(res_r2)

        cir_parms_final = np.array([res.x[0], res.x[1], res_r])
        #print(cir_parms_final, self.knownR)
        return cir_parms_final

    def draw_circle(self, img, n, t, window_str, thickness=2, elp_pts = None):
        if len(img.shape) < 3:
            imgT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            imgT = np.copy(img)

        R = self.get_Rotation(n)

        sita = np.linspace(0, np.pi * 2, 100)
        px = np.cos(sita) * self.knownR
        py = np.sin(sita) * self.knownR
        pz = np.zeros_like(px)
        pn = np.array([[0, 0, 0], [0, 0, self.knownR]])

        pt_cameraframe = np.matmul(R, np.array([px, py, pz])).transpose() + t
        pn_cameraframe = np.matmul(R, pn.transpose()).transpose() + t

        pt_imageframe = np.matmul(self.K, pt_cameraframe.transpose())
        pt_imageframe = (pt_imageframe[:-1, :] / pt_imageframe[-1, :]).transpose()

        pn_imageframe = np.matmul(self.K, pn_cameraframe.transpose())
        pn_imageframe = (pn_imageframe[:-1, :] / pn_imageframe[-1, :]).transpose()

        for pst, ped in zip(pt_imageframe[:-1, :], pt_imageframe[1:, :]):
            pst = np.round(pst).astype(dtype=np.int)
            ped = np.round(ped).astype(dtype=np.int)
            cv2.line(imgT, (pst[0], pst[1]), (ped[0], ped[1]), (0, 255, 0), thickness)

        pst = np.round(pn_imageframe[0, :]).astype(dtype=np.int)
        ped = np.round(pn_imageframe[1, :]).astype(dtype=np.int)
        cv2.circle(imgT, (pst[0], pst[1]), 3, (0, 0, 255), thickness)
        cv2.line(imgT, (pst[0], pst[1]), (ped[0], ped[1]), (0, 0, 255), thickness)

        if elp_pts is not None:
            for pt in elp_pts:
                cv2.circle(imgT, (pt[1], pt[0]), 2, (255, 0, 0), thickness)

        cv2.imshow(window_str, imgT)
        cv2.waitKey()

    def cost_perspective_pixels2plane(self, pts_imageframe, circle_norm, circle_pose):
        px, py, R, t = self.perspective_pixels2circle(pts_imageframe, circle_norm, circle_pose)

        err_abs = np.abs(np.sqrt(np.square(px) + np.square(py)) - self.knownR)


        #err_abs = np.abs(np.sqrt(px * px + py * py) - self.knownR)

        #err_mean =


        return np.sum(err_abs)

    @classmethod
    def corrected_norm(cls, init_norm, eta, xi):
        R = cls.get_Rotation(init_norm)
        Re = cls.get_error_Rotation(eta, xi)
        circle_norm = np.matmul(R, Re)[:, 2]
        return circle_norm

    def perspective_circle2image(self, circle_norm, circle_pos):
        Rwc = self.get_Rotation(circle_norm)

        circle_pos = np.array([circle_pos])
        xyzo = np.matmul(Rwc.transpose(), -circle_pos.transpose()).transpose()
        xo, yo, zo = xyzo[0]

        U = np.array([[zo * zo, 0, -xo * zo],
                      [0, zo * zo, -yo * zo],
                      [-xo * zo, -yo * zo, xo * xo + yo * yo - self.knownR * self.knownR]])
        RwcK_inv = np.matmul(Rwc.transpose(), np.linalg.inv(self.K))
        C = np.matmul(np.matmul(RwcK_inv.transpose(), U), RwcK_inv)
        # print(xo, yo, zo)
        X1, X2, N1, N2 = self.GetSingleCirclePose(C)

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


    def __call__(self, depth, pts_circleplane_pixelframe, pts_elprim_pixelframe):
        # Step 1: Plane Fitting, obtain circle norm
        pts_cameraframe = self.transform_Pixel2CameraFrame(pts_circleplane_pixelframe, depth)
        init_circle_norm, d, err, pts_center = self.fit_plane(pts_cameraframe)

        # Step 2: Fit space circle, obtain circle pos
        px, py, R, t = self.perspective_pixels2plane(pts_elprim_pixelframe, init_circle_norm, d)


        cir_parms = self.fit_circle(px, py)
        cir_pos = np.array([[cir_parms[0], cir_parms[1], 0]]).transpose()
        init_circle_pos = (np.matmul(R, cir_pos).transpose() + t)[0]

        #print(init_circle_pos, init_circle_norm)
        #exit()

        #print(cir_pos)
        # plt.plot(px, py, 'r', lw=2, label="pxy")
        # plt.plot(cir_pos[0], cir_pos[1], 'ob', lw=2, label="pose")
        # plt.plot(cir_pos[0] + self.knownR, cir_pos[1], 'ob', lw=2, label="pose")
        # plt.plot(cir_pos[0] - self.knownR, cir_pos[1], 'ob', lw=2, label="pose")
        # plt.plot(cir_pos[0], cir_pos[1] + self.knownR, 'ob', lw=2, label="pose")
        # plt.plot(cir_pos[0], cir_pos[1] + self.knownR, 'ob', lw=2, label="pose")
        # plt.plot(cir_pos[0] - self.knownR, cir_pos[1] - self.knownR, 'ob', lw=2, label="pose")
        # plt.plot(cir_pos[0] + self.knownR, cir_pos[1] + self.knownR, 'ob', lw=2, label="pose")
        # plt.plot(cir_pos[0] - self.knownR, cir_pos[1] + self.knownR, 'ob', lw=2, label="pose")
        # plt.plot(cir_pos[0] + self.knownR, cir_pos[1] - self.knownR, 'ob', lw=2, label="pose")
        # plt.axis('equal')
        # plt.show()
        # print(cir_parms)
        # exit()


        # Step 3: Set pose bound
        bounds = ([0, -np.pi, -self.knownR, -self.knownR, -self.knownR],
                   [10. / 180. * np.pi, np.pi, self.knownR, self.knownR, self.knownR])
        #bounds = ([0, -self.knownR, -self.knownR, -self.knownR, -self.knownR],
        #          [10. / 180. * np.pi, self.knownR, self.knownR, self.knownR, self.knownR])

        x0 = np.array([0. / 180. * np.pi, 0. / 180. * np.pi, 0, 0, 0])
        log_x = []

        # Step 4: Optimize Pose Cost
        init_circle_norm = np.array([0.73247912,0.10196938,- 0.67310964])
        init_circle_pos = np.array([-0.04522788,- 0.07558966, 0.68845102])

        def residual(x):
            log_x.append(x)
            xi = x[0]
            eta = x[1]
            err_pos = x[2:]
            R = self.get_Rotation(init_circle_norm)
            Re = self.get_error_Rotation(eta, xi)
            circle_norm = np.matmul(R, Re)[:, 2]
            circle_pos = init_circle_pos + err_pos
            cost = self.cost_perspective_pixels2plane(pts_elprim_pixelframe, circle_norm, circle_pos)



            #print(cost, xi/np.pi*180., circle_norm, circle_pos)
            return cost

        res = least_squares(residual, x0, verbose=1, bounds=bounds, max_nfev=10000)

        corrected_norm = self.corrected_norm(init_circle_norm, res.x[0], res.x[1])
        corrected_pos = init_circle_pos + res.x[2:]

        # Step 5: Final Pose Selection
        print(res.x)
        final_norm, final_pos = self.best_pose_selection(corrected_norm, corrected_pos, pts_cameraframe)

        return {'init_pose': (init_circle_norm, init_circle_pos),
                'refine_pose': (corrected_norm, corrected_pos),
                'final_pose': (final_norm, final_pos),
                'log_refine': log_x}
