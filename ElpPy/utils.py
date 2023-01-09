from turtle import shape
import numpy as np
import cv2
from numpy.lib.npyio import load
from ElpPy.lib import pyEllipseTools as elppy
import random

def makeContinueCppData(data:np.ndarray, dtype):
    if data.dtype != dtype or data.flags['C_CONTIGUOUS']==False:
        usage_data = np.ascontiguousarray(data.astype(dtype))
    else:
        usage_data = data
    return usage_data

# 注意这里面所有的角度都是弧度，且为半长短轴
class GeneralEllipse(object):
    # 输入椭圆的参数，可以是形状，可以是参数方程，输入参数时候给上对应的参数类型
    def __init__(self, elp_data = None, elp_type = None, is_real = False, gelp_dict = None) -> None:
        super().__init__()
        
        self.is_real = is_real
        
        self.type_shape_mat = 'shape_matrix'
        self.type_shape_img = 'shape_image'
        self.type_equ_mat = 'equ_matrix'
        self.type_equ_img = 'equ_image'
        
        self.all_elp_type = {self.type_shape_mat: None, 
                             self.type_shape_img: None, 
                             self.type_equ_mat: None, 
                             self.type_equ_img: None}
        
        if elp_data is None and elp_type is None and is_real is False and gelp_dict is None:
            return
        
        
        if gelp_dict is not None: # 存在字典数据，直接导入
            self.loadData(gelp_dict)
            return 
        
        # print(list(self.all_elp_type.keys()))
        assert(elp_type in list(self.all_elp_type.keys()))
        self.all_elp_type[elp_type] = np.array(elp_data).astype(np.float)
        self.usage_type = elp_type
        self.is_real = True
        
        self.recoverFullPose()
        
    def loadData(self, gelp_dict):
        self.all_elp_type[self.type_shape_mat] = np.array(gelp_dict[self.type_shape_mat])
        self.all_elp_type[self.type_shape_img] = np.array(gelp_dict[self.type_shape_img])
        self.all_elp_type[self.type_equ_mat] = np.array(gelp_dict[self.type_equ_mat])
        self.all_elp_type[self.type_equ_img] = np.array(gelp_dict[self.type_equ_img])
        self.is_real = True
    
    def getDict(self):
        assert(self.is_real)
        
        return self.all_elp_type
        
    def checkValid(self):
        assert(self.is_real)
        elp_shape = self.all_elp_type[self.type_shape_img]
        if elp_shape[2] > 0 and elp_shape[3] > 0:
            return True
        else:
            return False
        
    def ellipse_shape_mat(self) -> np.ndarray:
        assert(self.is_real)
        return self.all_elp_type[self.type_shape_mat]
    
    def ellipse_shape_img(self, make_continous = False) -> np.ndarray:
        assert(self.is_real)
        
        elp = self.all_elp_type[self.type_shape_img]
        if make_continous:
            if elp.dtype != np.float or elp.flags['C_CONTIGUOUS']==False:
                usage_elp = np.ascontiguousarray(elp.astype(np.float))
            else:
                usage_elp = elp
        else:
            usage_elp = elp
        return usage_elp
    
    def ellipse_equation_mat(self) -> np.ndarray:
        assert(self.is_real)
        return self.all_elp_type[self.type_equ_mat]
    
    def ellipse_equation_img(self) -> np.ndarray:
        assert(self.is_real)
        return self.all_elp_type[self.type_equ_img]
    
    
    @classmethod
    def equ2mat33(cls, equ:np.ndarray) -> np.ndarray:
        assert(len(equ.shape) == 1 and equ.shape[0] == 6)
        return np.array([[equ[0], equ[1], equ[3]],
                         [equ[1], equ[2], equ[4]],
                         [equ[3], equ[4], equ[5]]], dtype=np.float)
        
    
    
    def recoverFullPose(self):
        assert(self.is_real)
        usage_data = self.all_elp_type[self.usage_type]
        if self.usage_type == self.type_shape_mat or self.usage_type == self.type_shape_img:
            data_shape_image = np.array([usage_data[1], usage_data[0], usage_data[3], 
                                         usage_data[2], -usage_data[4]], dtype=np.float)
            if self.usage_type == self.type_shape_mat:
                self.all_elp_type[self.type_shape_img] = data_shape_image
            else:
                self.all_elp_type[self.type_shape_mat] = data_shape_image
            # print(type(self.all_elp_type[self.type_shape_mat]))
            self.all_elp_type[self.type_equ_mat] = self.shape2equ(self.all_elp_type[self.type_shape_mat])
            self.all_elp_type[self.type_equ_img] = self.shape2equ(self.all_elp_type[self.type_shape_img])
        else:
            data_shape = self.equ2shape(usage_data)
            data_shape_cvt = np.array([data_shape[1], data_shape[0], data_shape[3], 
                                         data_shape[2], -data_shape[4]], dtype=np.float)
            # print(data_shape, data_shape_cvt)
            if self.usage_type == self.type_equ_img:
                self.all_elp_type[self.type_shape_img] = data_shape
                self.all_elp_type[self.type_shape_mat] = data_shape_cvt
                self.all_elp_type[self.type_equ_mat] = self.shape2equ(self.all_elp_type[self.type_shape_mat])
            else:
                self.all_elp_type[self.type_shape_mat] = data_shape
                self.all_elp_type[self.type_shape_img] = data_shape_cvt
                self.all_elp_type[self.type_equ_img] = self.shape2equ(self.all_elp_type[self.type_shape_img])
        
    @classmethod
    def shape2equ(cls, elp:np.ndarray) -> np.ndarray:
        if elp.dtype != np.float or elp.flags['C_CONTIGUOUS']==False:
            usage_elp = np.ascontiguousarray(elp.astype(np.float))
        else:
            usage_elp = elp
        return elppy.pyELPShape2Equation(usage_elp)
    
    @classmethod
    def equ2shape(cls, equ:np.ndarray) -> np.ndarray:
        if equ.dtype != np.float or equ.flags['C_CONTIGUOUS']==False:
            usage_elp = np.ascontiguousarray(equ.astype(np.float))
        else:
            usage_elp = equ
        return elppy.pyELPEquation2Shape(usage_elp)
    
    def getArea(self):
        assert(self.is_real)
        usage_shape_data = self.all_elp_type[self.type_shape_img]
        return usage_shape_data[2] * usage_shape_data[3] * np.pi
    
    def drawEllipse(self, imgC, color, thickness = 2, center_offset = None):
        assert(self.is_real)
        usage_shape_data = self.all_elp_type[self.type_shape_img]
        
        if usage_shape_data[2] <= 0 or usage_shape_data[3] <= 0:
            return False
        
        if center_offset is not None:
            draw_shape = ((usage_shape_data[0] + center_offset[0], usage_shape_data[1] + center_offset[1]), 
                      (usage_shape_data[2] * 2, usage_shape_data[3] * 2),
                      usage_shape_data[4]/np.pi * 180.0 )
        else:
            draw_shape = ((usage_shape_data[0], usage_shape_data[1]), 
                        (usage_shape_data[2] * 2, usage_shape_data[3] * 2),
                        usage_shape_data[4]/np.pi * 180.0 )
        cv2.ellipse(imgC, draw_shape, color, thickness=thickness)
        return True
        
    @classmethod
    def calRangeOfY(cls, elp: np.ndarray) -> np.ndarray:
        if elp.dtype != np.float or elp.flags['C_CONTIGUOUS']==False:
            usage_elp = np.ascontiguousarray(elp.astype(np.float))
        else:
            usage_elp = elp
        
        xmin, xmax, ymin, ymax = elppy.pyCalculateRangeOfY(usage_elp)
        return np.array([xmin, xmax, ymin, ymax], dtype=np.float)
    
    @classmethod
    def calRangeAtY(cls, elp: np.ndarray, idx_y: np.float) -> list:
        if elp.dtype != np.float or elp.flags['C_CONTIGUOUS']==False:
            usage_elp = np.ascontiguousarray(elp.astype(np.float))
        else:
            usage_elp = elp
            
        xmin, xmax = elppy.pyCalculateRangeAtY(usage_elp, idx_y)
        
        return [xmin, xmax]
    
    # 计算椭圆内部像素点 N*2
    def inner_points(self, irows: int, icols: int):
        assert(self.is_real)
        elp = self.all_elp_type[self.type_shape_img]
        equ = self.all_elp_type[self.type_equ_img]
        
        xmin, xmax, ymin, ymax = elppy.pyCalculateRangeOfY(elp)
        ymin = np.max([0, np.floor(ymin)])
        ymax = np.min([irows - 1, np.ceil(ymax)])
        
        pts_mask = []
        for idx_y in range(int(ymin), int(ymax) + 1):
            x_min_max = elppy.pyCalculateRangeAtY(equ, np.float64(idx_y))
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
    
    def get_center_img(self):
        assert(self.is_real)
        elp_shape = self.ellipse_shape_img()
        return elp_shape[0:2]
    
    def GenerateElpData(self, pts_num = 200, format_img = True):
        assert(self.is_real)
        if format_img:
            elp_shape = self.ellipse_shape_img()
        else:
            elp_shape = self.ellipse_shape_mat()
        
        sita = np.linspace(0, 2 * np.pi, pts_num)
    
        xc, yc, R, r, theta = elp_shape
    
        rot = np.array([[np.math.cos(theta), -np.math.sin(theta)],
                    [np.math.sin(theta), np.math.cos(theta)]], dtype=np.float)
    
        # print(sita)
        x0 = R * np.cos(sita)
        y0 = r * np.sin(sita)
    
        XY = np.matmul(rot, np.vstack([x0, y0]))
    
        x = XY[0, :] + xc
        y = XY[1, :] + yc
    
        return x, y
    
    # [u,v] 输入的是标准的图像坐标系
    def calDistance(self, pts:np.ndarray)->np.ndarray:
        assert(self.is_real)
        N, d = pts.shape
        assert(d == 2)
        
        parms = self.all_elp_type[self.type_equ_img]
        cx, cy = self.all_elp_type[self.type_shape_img][0:2]
        
        # parms = self.all_elp_type[self.type_equ_mat]
        # cx, cy = self.all_elp_type[self.type_shape_mat][0:2]
        
        px = pts[:, 0]
        py = pts[:, 1]
        
        dx = px - cx
        dy = py - cy
        dx2 = dx * dx
        dy2 = dy * dy;
        ldxy = np.sqrt(dx2 + dy2)
        
        
        pc1 = parms[3] + parms[0] * cx + parms[1] * cy
        pc2 = parms[4] + parms[1] * cx + parms[2] * cy
        pc3 = parms[0] * cx * cx + 2 * parms[1] * cx * cy + 2 * parms[3] * cx + parms[2] * cy * cy + 2 * parms[4] * cy + parms[5]
        
        
        p1 = parms[0] * dx2 + parms[2] * dy2 + 2 * parms[1] * dx * dy
        p2 = pc1 * dx + pc2 * dy
        tmp = np.sqrt(np.abs(p2 * p2 - p1 * pc3));
        
        t1 = (-p2 + tmp) / p1
        t2 = (-p2 - tmp) / p1
        
        # print('t1', t1)
        # print('t2', t2)
        
        dst1 = np.abs(t1 - 1) * ldxy
        dst2 = np.abs(t2 - 1) * ldxy
        
        dst1[t1 < 0] = dst2[t1 < 0]
        
        # print('dst1', dst1)
        
        return dst1
        
    



class GeneralLine(object):
    def __init__(self, pts = None, equ = None) -> None:
        super().__init__()
        
        self.is_real = False
        if pts is not None or equ is not None:
            self.is_real = True
        
        self.pts = pts
        self.line_np = None
        
        self.equ = equ
        
        if pts is not None:
            N, d = pts.shape
            assert(N >= 2 and d == 2)
            # print('pts', pts)
            fitted_line = np.array(cv2.fitLine(pts, cv2.DIST_L2, 0, 1e-2, 1e-2)).transpose()[0]
            # print('fitted_line', fitted_line)
            line_p = fitted_line[1]
            line_q = -fitted_line[0]
            line_k = -fitted_line[1] * fitted_line[2] + fitted_line[0] * fitted_line[3]
            self.equ = np.array([line_p, line_q, line_k])
            self.line_np = fitted_line

    
    def loadData(self, gline_dict):
        key_list = list(gline_dict.keys())
        if 'pts' in key_list:
            self.pts = np.array(gline_dict['pts'])
        
        if 'equ' in key_list:
            self.equ = np.array(gline_dict['equ'])
            
        if 'line_np' in key_list:
            self.line_np = np.array(gline_dict['line_np'])
            
        self.is_real = True
    
    def getDict(self):
        assert(self.is_real)
        
        line_dict = {}
        if self.pts is not None:
            line_dict['pts'] = self.pts
        if self.equ is not None:
            line_dict['equ'] = self.equ
        if self.line_np is not None:
            line_dict['line_np'] = self.line_np
            
        return line_dict
    
    
    # 返回每个点在直线上的投影。相对于拟合之心的中心点坐标
    def perspectPoints2Line(self, pts:np.ndarray)->np.ndarray:
        assert(self.is_real)
        N, d = pts.shape
        assert(N >= 1 and d == 2)
        
        n = self.line_np[0: 2]
        O = self.line_np[2:]
        
        OP = pts - O
        
        t = np.sum(OP * n, axis=1) / np.linalg.norm(n)
        
        return t
        
        
    
    def GenerateLineData(self, img_width = -1, img_height = -1, pts_num = 200):
        a, b, c = self.equ
        if self.pts is None:
            assert(img_width > 0 and img_height > 0)
            if abs(b) > 1e-3:
                px = np.linspace(0, img_width, pts_num)
                py = (-a * px - c) / b
            else:
                py = np.linspace(0, img_height, pts_num)
                px = (-b * py - c) / a
                
            idx_inner = np.argwhere(0 <= px <= img_width and 0 <= py <= img_height)
            px = px[idx_inner]
            py = py[idx_inner]
        else: # 存在采样点
            t = self.perspectPoints2Line(self.pts)
            t_min = np.min(t)
            t_max = np.max(t)
            
            # print(t)
            
            sample_t = np.linspace(t_min, t_max, num = pts_num)
            n = self.line_np[0:2]
            O = self.line_np[2:]
            n /= np.linalg.norm(n)
            
            sample_py = np.vstack([sample_t, sample_t]).transpose() * n + O
            
            # print(n)
            # print(sample_t)
            # print(O)
            
            px = sample_py[:, 0]
            py = sample_py[:, 1]
            
        return px, py
    
    def calDistance(self, pts:np.ndarray)->np.ndarray:
        pu = pts[:, 0]
        pv = pts[:, 1]
        
        equ = self.equ
        dist = np.abs(equ[0] * pu + equ[1] * pv + equ[2]) / np.linalg.norm(equ[:2])
        return dist
    
class GeneralLineSegment(GeneralLine):
    def __init__(self, pt1:np.ndarray, pt2:np.ndarray):
        pts = np.vstack([pt1, pt2])
        super().__init__(pts=pts)
        
        self.pt1 = pt1
        self.pt2 = pt2
        self.n12 = pt2 - pt1
        self.ln12 = np.linalg.norm(self.n12)
        
    def calPerspectiveScale(self, pts):
        nl = pts - self.pt1
        pl = np.dot(nl, self.n12) / self.ln12
        return pl / self.ln12



def MakeCPPContinues(data:np.ndarray):
    if data.dtype != np.float or data.flags['C_CONTIGUOUS']==False:
        usage_data = np.ascontiguousarray(data.astype(np.float))
    else:
        usage_data = data
    return usage_data
        
# S是拟合矩阵，C是约束矩阵
def GeneralDirectLeastSquare(S:np.array, C:np.array):
    if S.dtype != np.float or S.flags['C_CONTIGUOUS']==False:
        usage_S = np.ascontiguousarray(S.astype(np.float))
    else:
        usage_S = S
        
    if C.dtype != np.float or C.flags['C_CONTIGUOUS']==False:
        usage_C = np.ascontiguousarray(C.astype(np.float))
    else:
        usage_C = C
        
    dataX, err = elppy.pyGeneralDirectLeastSquare(usage_S, usage_C)
    
    return dataX, err


def ElliFit(pts: np.ndarray, elp_type:str, fitCircle = False)->GeneralEllipse:
    N, d = pts.shape
    assert(N > 5 and d == 2)
    
    if pts.dtype != np.float or pts.flags['C_CONTIGUOUS']==False:
        usage_pts = np.ascontiguousarray(pts.astype(np.float))
    else:
        usage_pts = pts
    
    if fitCircle:
        fitres = elppy.pyFitCircle(usage_pts)
    else:
        fitres = elppy.pyFitEllipse(usage_pts)
    
    if fitres[2] < 0 or fitres[3] < 0:
        return None
    else: 
        return GeneralEllipse(fitres, elp_type)




class GeneralCamera(object):
    def __init__(self, fx:float, fy:float, cx:float, cy:float, dist_model = None, coeffs = None) -> None:
        super().__init__()
        
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.dist_model = dist_model
        self.coeffs = coeffs
        
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float)
    
    @classmethod
    def loadDict(cls, gcam_dict):
        fx = gcam_dict['fx']
        fy = gcam_dict['fy']
        cx = gcam_dict['cx']
        cy = gcam_dict['cy']
        dist_model = None
        coeffs = None
        if 'dist_model' in gcam_dict.keys():
            dist_model = gcam_dict['dist_model']
            coeffs = gcam_dict['coeffs']
        return GeneralCamera(fx, fy, cx, cy, dist_model, coeffs)
    
    def getDict(self):
        usage_dict = {}
        usage_dict['fx'] = self.fx
        usage_dict['fy'] = self.fy
        usage_dict['cx'] = self.cx
        usage_dict['cy'] = self.cy
        usage_dict['K'] = self.K
        
        usage_dict['dist_model'] = self.dist_model
        usage_dict['coeffs'] = self.coeffs
        
        return usage_dict
    
    @classmethod
    def returnOne(cls, load_dict):
        return cls.loadDict(load_dict)
        # return GeneralCamera(load_dict['fx'], load_dict['fy'], load_dict['cx'], load_dict['cy'])
    
    def cameraK(self) -> np.ndarray:
        return self.K
    
    
    def cvtpts_cam2img(self, pts: np.ndarray):
        N, d = pts.shape
        assert(d == 3)
        uvz = np.matmul(self.K, pts.transpose())
        
        z = uvz[2, :]
        mask = z > 0
        u = uvz[0, :]
        v = uvz[1, :]
        z = z[mask]
        u = u[mask]
        v = v[mask]
        u /= z
        v /= z
        return np.vstack([u,v]).transpose()
    
    def project_point_to_pixel(self, points: np.ndarray):
        
        xs = points[:, 0] / points[:, 2]
        ys = points[:, 1] / points[:, 2]
        
        if self.dist_model == 'distortion.brown_conrady':
            r2s = xs * xs + ys * ys
            fs = 1 + self.coeffs[0] * r2s + self.coeffs[1] * r2s * r2s + self.coeffs[4] * r2s * r2s * r2s
            xfs = xs * fs
            yfs = ys * fs
            dxs = xfs + 2 * self.coeffs[2] * xs * ys + self.coeffs[3] * (r2s + 2 * xs * xs)
            dys = yfs + 2 * self.coeffs[3] * xs * ys + self.coeffs[2] * (r2s + 2 * ys * ys)
            xs = dxs
            ys = dys
        
        pu = xs * self.fx + self.cx
        pv = ys * self.fy + self.cy
        return np.vstack([pu, pv]).transpose()
            
    
    # pts: 双列，xy格式坐标
    def deproject_pixel_to_point(self, pts: np.ndarray, depths:np.ndarray = None)-> np.ndarray:
        # print(pts.shape)
        xs = (pts[:, 0] - self.cx) / self.fx
        ys = (pts[:, 1] - self.cy) / self.fy
        xo = np.copy(xs)
        yo = np.copy(ys)
        
        # print(xs.shape, ys.shape)
        
        # print(self.dist_model)
        # print(self.dist_model == 'distortion.brown_conrady')
        # exit()
        if self.dist_model == 'distortion.brown_conrady':
            # need to loop until convergence 
            # 10 iterations determined empirically
            for i in range(10):
                r2s = xs * xs + ys * ys
                icdists = 1.0 / (1.0 + ((self.coeffs[4] * r2s + self.coeffs[1]) * r2s + self.coeffs[0]) * r2s)
                delta_xs = 2 * self.coeffs[2] * xs * ys + self.coeffs[3] * (r2s + 2 * xs * xs)
                delta_ys = 2 * self.coeffs[3] * xs * ys + self.coeffs[2] * (r2s + 2 * ys * ys)
                xs = (xo - delta_xs) * icdists
                ys = (yo - delta_ys) * icdists
                
                # print(r2s.shape, icdists.shape, delta_xs.shape, delta_ys.shape, xs.shape, ys.shape)
        
        if depths is not None:
            pts3d = np.vstack([depths[:, 0] * xs, depths[:, 0] * ys, depths[:, 0]]).transpose()
        else:
            pts3d = np.vstack([xs, ys, np.ones_like(xs)]).transpose()
        # print(pts3d.shape)
        # exit()
        return pts3d


class GeneralSpacialCircle(object):
    def __init__(self, cnorm: np.ndarray, cloc: np.ndarray, cr:np.float) -> None:
        super().__init__()
        
        self.cnorm = cnorm / np.linalg.norm(cnorm)
        # if self.cnorm[2] > 0:
        #     self.cnorm *= -1
        self.cloc = cloc
        self.cr = cr
     
    def getDict(self):
        res = {}
        res['cnorm'] = self.cnorm
        res['cloc'] = self.cloc
        res['cr'] = self.cr
        return res
    
    @classmethod
    def returnOne(cls, load_dict):
        # print(load_dict)
        cnorm = np.array(load_dict['cnorm'])
        cloc = np.array(load_dict['cloc'])
        cr = load_dict['cr']
        
        return GeneralSpacialCircle(cnorm, cloc, cr)
    
    def loadDict(self, load_dict):
        self.cnorm = load_dict['cnorm']
        self.cloc = load_dict['cloc']
        self.cr = load_dict['cr']
    
    def generateCircularSamples(self, sample_num = 200):
        R = get_norm_Rotation(self.cnorm)
        sita = np.linspace(0, 2 * np.pi, sample_num)
        xw = np.cos(sita) * self.cr
        yw = np.sin(sita) * self.cr
        zw = np.zeros_like(xw)
        xyzw = np.vstack([xw, yw, zw])
        xyzc = np.matmul(R, xyzw).transpose() + self.cloc
        return xyzc
    
    def generateNormal(self):
        R = get_norm_Rotation(self.cnorm)
        pts = np.array([[0, 0, 0], [0, 0, self.cr]]).transpose()
        # print('generateNormal:pts', pts)
        xyzc = np.matmul(R, pts).transpose() + self.cloc
        return xyzc
        
        
        
        
        
        
        
def trans2Dpixels_to_3Dpoints(gcam: GeneralCamera, pts:np.ndarray, depth: np.ndarray, depth_factor: np.float):
    assert(pts.shape[1] == 2)
    
    usage_depth = depth[pts[:, 0], pts[:, 1]]
    valid_idx = np.argwhere(usage_depth > 0)
    valid_num = len(valid_idx)
    # usage_depth = usage_depth[usage_depth > 0]
    usage_depth = np.reshape(usage_depth[valid_idx], (valid_num, 1)) * depth_factor
    
    # print('min, max', np.min(usage_depth), np.max(usage_depth))
    # print(usage_depth)
    if gcam is None:
        return np.hstack([pts[valid_idx, [1, 0]], usage_depth])
    else:
        # print(pts[valid_idx, [1, 0]].shape, usage_depth.shape)
        # exit()
        usage_pts = np.array(pts[valid_idx, [1, 0]])
        # print(usage_pts.shape, valid_idx.shape)
        # exit()
        pts3d = gcam.deproject_pixel_to_point(usage_pts, usage_depth)
        # print(pts3d.shape)
        # exit()
        return pts3d
        fx = gcam.fx
        fy = gcam.fy
        cx = gcam.cx
        cy = gcam.cy
        
        u = np.reshape(pts[valid_idx, 1], (valid_num, 1))
        v = np.reshape(pts[valid_idx, 0], (valid_num, 1))
        
        
        px = (u - cx) / fx * usage_depth
        py = (v - cy) / fy * usage_depth
        
        return np.hstack([px, py, usage_depth])
    


def norm_dist(n1:np.ndarray, n2: np.ndarray)->float:
    # cos_theta = np.abs(np.dot(n1, n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    cos_theta = (np.dot(n1, n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    # print('cos_theta', cos_theta)
    diff_angle = np.math.acos(cos_theta) / np.pi * 180.0
    return diff_angle

def position_dist(p1:np.ndarray, p2:np.ndarray)->float:
    dist = p1 - p2
    return np.linalg.norm(dist)

def radius_dist(r1, r2):
    return abs(r1 - r2)

def get_norm_Rotation(n:np.ndarray):
    n /= np.linalg.norm(n)
    n1, n2, n3 = n
    
    t = np.math.sqrt(n1 * n1 + n2 * n2)
    invt = 1.0 / t
    
    R = np.array([[-n2 * invt, -n1 * n3 * invt, n1],
                  [n1 * invt, -n2 * n3 * invt, n2],
                  [0, t, n3]], dtype=np.float)
    return R


# 输入深度图单位为m，double类型
def depth_colorizer(depth:np.ndarray):
    assert(len(depth.shape) == 2)
    
    if depth.dtype != np.float or depth.flags['C_CONTIGUOUS']==False:
        # print('not continues')
        usage_depth = np.ascontiguousarray(depth.astype(np.float))
    else:
        usage_depth = depth
    
    usage_depth[usage_depth > 60000] = 0.0
    usage_depth *= 1000
    usage_depth = usage_depth.astype(dtype='uint16')
    
    return cv2.cvtColor(elppy.pyDepthColorizer(usage_depth), cv2.COLOR_RGB2BGR) 


def calCannyThreshold(imgG: np.ndarray):
    assert(len(imgG.shape) == 2)
    usage_imgG = makeContinueCppData(imgG, np.uint8)
    low, high = elppy.pycalCannyThreshold(usage_imgG)
    return low, high


def perspectCircular2Image(gcam:GeneralCamera, gscir:GeneralSpacialCircle)->GeneralEllipse:
    Rwc = get_norm_Rotation(gscir.cnorm).transpose()
    K = gcam.cameraK()
    
    # print(np.matmul(Rwc, np.array([gscir.cloc]).transpose()).transpose())
    xo, yo, zo = np.matmul(Rwc, -np.array([gscir.cloc]).transpose()).transpose()[0]
    # print('xo, yo, zo', xo, yo, zo)
    # xo, yo, zo = gscir.cloc
    U = np.array([[zo * zo, 0, -xo * zo],
                  [0, zo * zo, -yo * zo],
                  [-xo * zo, -yo * zo, xo * xo + yo * yo - gscir.cr * gscir.cr]])
    L = np.matmul(Rwc, np.linalg.inv(K))
    C = np.matmul(np.matmul(L.transpose(), U), L)
    
    equ = np.array([C[0, 0], C[0, 1], C[1, 1], C[0, 2], C[1, 2], C[2, 2]])
    
    k = equ[0] * equ[2] - equ[1] * equ[1]
    
    equ /= np.math.sqrt(abs(k))
    
    # print(equ)
    return GeneralEllipse(elp_data=equ, elp_type='equ_image')


def drawPerspectiveCircle(imgC: np.ndarray, gelp:GeneralEllipse, 
                          elliptical_pixels = None, square_pixels = None, cooperate_line = None,
                          other_pts = None, mask_pts = None):
    # # 生成圆的采样点
    # xyzc = gscir.generateCircularSamples()
    # pts = gcam.cvtpts_cam2img(xyzc)
    # for each_pt in pts:
    #     each_pt = np.round(each_pt).astype(int)
    #     cv2.circle(imgC, (each_pt[0], each_pt[1]), 2, (0, 0, 255))
    
    if mask_pts is not None:
        if len(imgC.shape) == 2:
            imgC[mask_pts[:, 1], mask_pts[:, 0]] -= 50
        else:
            imgC[mask_pts[:, 1], mask_pts[:, 0], 1] = 0
    
    if elliptical_pixels is not None:
        for each_pt in elliptical_pixels:
            cv2.circle(imgC, (int(each_pt[0] + 0.5), int(each_pt[1] + 0.5)), 1, (0 , 255, 0), 1)
    
    usage_elp = gelp
    
    if usage_elp.checkValid():
        usage_elp.drawEllipse(imgC, (0, 0, 255), 2)
    else:
        return
    
    # print(gelp.all_elp_type)
    
    # tgelp = ElliFit(pts, 'shape_image')
    
    # print(tgelp.all_elp_type)
    
    # gelp.drawEllipse(imgC, (0, 0, 255), 2)
    # # 绘制法向向量
    # pts_norm = gscir.generateNormal()
    # pts = gcam.cvtpts_cam2img(pts_norm)
    # # print('pts_norm', pts_norm, 'pts', pts)
    # if pts.shape[0] >= 2:
    #     cv2.circle(imgC, (int(pts[0, 0] + 0.5), int(pts[0, 1] + 0.5)), 3, (0 , 255, 0), 2)
    #     cv2.line(imgC, (int(pts[0, 0] + 0.5), int(pts[0, 1] + 0.5)), 
    #              (int(pts[1, 0] + 0.5), int(pts[1, 1] + 0.5)), (0, 0, 255), 2)
    
    
    if square_pixels is not None and len(square_pixels) > 0:
        for each_pts in square_pixels:
            # print(each_pts)
            cv2.circle(imgC, (int(each_pts[0] + 0.5), int(each_pts[1] + 0.5)), 3, (0 , 255, 0), 2)
        # print(square_pixels)
        cv2.line(imgC, (int(square_pixels[0, 0] + 0.5), int(square_pixels[0, 1] + 0.5)), 
             (int(square_pixels[1, 0] + 0.5), int(square_pixels[1, 1] + 0.5)), (0, 0, 255), 2)
        
    if cooperate_line is not None and len(cooperate_line) > 0:
        pst = np.round(cooperate_line[0]).astype(int)
        ped = np.round(cooperate_line[1]).astype(int)
        cv2.line(imgC, (pst[0], pst[1]),  (ped[0], ped[1]), (0, 255, 0), 2)
        
    if other_pts is not None:
        for each_pt in other_pts:
            cv2.circle(imgC, (int(each_pt[0] + 0.5), int(each_pt[1] + 0.5)), 2, (0 , 0, 255), 2)

def drawCirclePose(imgC: np.ndarray, gcam: GeneralCamera, gscir: GeneralSpacialCircle, 
                   elliptical_pixels = None, square_pixels = None, cooperate_line = None,
                   other_pts = None, mask_pts = None, ingelp = None, ellipse_thickness = 2):
    
    # # 生成圆的采样点
    # xyzc = gscir.generateCircularSamples()
    # pts = gcam.cvtpts_cam2img(xyzc)
    # for each_pt in pts:
    #     each_pt = np.round(each_pt).astype(int)
    #     cv2.circle(imgC, (each_pt[0], each_pt[1]), 2, (0, 0, 255))
    
    if mask_pts is not None:
        if len(imgC.shape) == 2:
            imgC[mask_pts[:, 1], mask_pts[:, 0]] -= 50
        else:
            imgC[mask_pts[:, 1], mask_pts[:, 0], 2] = 0
    
    if elliptical_pixels is not None:
        for each_pt in elliptical_pixels:
            # cv2.circle(imgC, (int(each_pt[0] + 0.5), int(each_pt[1] + 0.5)), 1, (0 , 255, 0), 1)
            imgC[int(each_pt[1] + 0.5), int(each_pt[0] + 0.5), 0] = 0
            imgC[int(each_pt[1] + 0.5), int(each_pt[0] + 0.5), 1] = 255
            imgC[int(each_pt[1] + 0.5), int(each_pt[0] + 0.5), 2] = 0
          
    if ingelp is not None:
        usage_elp = ingelp
    else:
        usage_elp = perspectCircular2Image(gcam, gscir)
    
    
    if usage_elp.checkValid():
        usage_elp.drawEllipse(imgC, (0, 0, 255), ellipse_thickness)
        # 绘制法向向量
        pts_norm = gscir.generateNormal()
        pts = gcam.cvtpts_cam2img(pts_norm)
        # print('pts_norm', pts_norm, 'pts', pts)
        if pts.shape[0] >= 2:
            cv2.circle(imgC, (int(pts[0, 0] + 0.5), int(pts[0, 1] + 0.5)), 3, (0 , 255, 0), 2)
            cv2.line(imgC, (int(pts[0, 0] + 0.5), int(pts[0, 1] + 0.5)), 
                    (int(pts[1, 0] + 0.5), int(pts[1, 1] + 0.5)), (0, 0, 255), 2)
    else:
        return
    
    # print(gelp.all_elp_type)
    
    # tgelp = ElliFit(pts, 'shape_image')
    
    # print(tgelp.all_elp_type)
    
    # gelp.drawEllipse(imgC, (0, 0, 255), 2)
    # # 绘制法向向量
    # pts_norm = gscir.generateNormal()
    # pts = gcam.cvtpts_cam2img(pts_norm)
    # # print('pts_norm', pts_norm, 'pts', pts)
    # if pts.shape[0] >= 2:
    #     cv2.circle(imgC, (int(pts[0, 0] + 0.5), int(pts[0, 1] + 0.5)), 3, (0 , 255, 0), 2)
    #     cv2.line(imgC, (int(pts[0, 0] + 0.5), int(pts[0, 1] + 0.5)), 
    #              (int(pts[1, 0] + 0.5), int(pts[1, 1] + 0.5)), (0, 0, 255), 2)
    
    
    if square_pixels is not None and len(square_pixels) > 0:
        for each_pts in square_pixels:
            # print(each_pts)
            cv2.circle(imgC, (int(each_pts[0] + 0.5), int(each_pts[1] + 0.5)), 3, (0 , 255, 0), 2)
        # print(square_pixels)
        cv2.line(imgC, (int(square_pixels[0, 0] + 0.5), int(square_pixels[0, 1] + 0.5)), 
             (int(square_pixels[1, 0] + 0.5), int(square_pixels[1, 1] + 0.5)), (0, 0, 255), 2)
        
    if cooperate_line is not None and len(cooperate_line) > 0:
        pst = np.round(cooperate_line[0]).astype(int)
        ped = np.round(cooperate_line[1]).astype(int)
        cv2.line(imgC, (pst[0], pst[1]),  (ped[0], ped[1]), (0, 255, 0), 2)
        
    if other_pts is not None:
        for each_pt in other_pts:
            cv2.circle(imgC, (int(each_pt[0] + 0.5), int(each_pt[1] + 0.5)), 2, (0 , 0, 255), 2)
        
        

def IoUEllipses(gelp1: GeneralEllipse, gelp2: GeneralEllipse):
    shape_elp1 = gelp1.ellipse_shape_img(make_continous=True)
    shape_elp2 = gelp2.ellipse_shape_img(make_continous=True)
    
    # print(shape_elp1)
    # print(shape_elp2)
    ratio = elppy.pyfasterCalculateOverlap(shape_elp1, shape_elp2)
    return ratio

# allgelp是一个元素类型为ELSDData的list # ELSD算法存在1个像素的偏移
def findMatchEllipse(gelp: GeneralEllipse, allelsdelp: list, T_elp, ellipse_pixel_offset = 1):
    is_find = False
    
    found_ellipticl_pts = None
    found_elp_data = None
    found_ellipse = None
    
    for each_det_elp in allelsdelp:
        tmp_pts = np.array(each_det_elp.regs)
        dist = gelp.calDistance(tmp_pts + ellipse_pixel_offset)
        # print(np.mean(dist), np.max(dist), np.min(dist), np.sum(dist > T_elp), len(dist))
        # print(np.min(dist))
        # np.mean(dist)
        
        if np.mean(dist) < T_elp:
            if found_ellipticl_pts is None or len(found_ellipticl_pts) < len(tmp_pts):
                found_ellipticl_pts = tmp_pts
                is_find = True
                found_elp_data = each_det_elp
    if not is_find:
        return None
    
    cx = found_elp_data.cx
    cy = found_elp_data.cy
    rx = found_elp_data.rx
    ry = found_elp_data.ry
    angle = found_elp_data.angle /180.0 * np.pi
    elp_data = np.array([cx, cy, rx, ry, angle])
    elp_type = 'shape_image'
    found_ellipse = GeneralEllipse(elp_data=elp_data, elp_type=elp_type)

    return found_ellipse, found_ellipticl_pts



# 查找Mask
def findMatchMask(gt_masks, det_masks, idx_gt):
    found_mask = None
    num_masks = det_masks.shape[0]
    for idx_masks in range(num_masks):
        usage_mask = det_masks[idx_masks, 0]
        # tmpimage = Image.fromarray(usage_mask)
        # tmpimage.show()
        # print(usage_mask.shape)
        # print(np.argwhere(usage_mask > 0))
        idxes = np.argwhere(usage_mask > 0)
        mask_px = idxes[:, 0]
        mask_py = idxes[:, 1]
        # mask_px, mask_py = np.argwhere(usage_mask > 0)
        idx = np.argwhere(gt_masks[mask_px, mask_py] == idx_gt + 1)
        if len(idx) > 3:
            if found_mask is None:
                found_mask = np.vstack([mask_py, mask_px]).transpose()
            else:
                if found_mask.shape[0] < len(idx):
                    found_mask = np.vstack([mask_py, mask_px]).transpose()
                    
    return found_mask

# 查找匹配的直线段
def findMatchLineSegment(gline, det_lsd, T_line):
    # 寻找直线段
    found_linesements = None
    usage_s_range = None

    glinesegment = gline
    for each_line in det_lsd:
        pt1 = each_line[:2]
        pt2 = each_line[2:]
        tpts = np.vstack([pt1, pt2])
        dist = glinesegment.calDistance(tpts)
        sper = glinesegment.calPerspectiveScale(tpts)
        if np.mean(dist) < T_line:
            s1, s2 = np.sort(sper)
            smin = np.max([0.0, s1])
            smax = np.min([1.0, s2])
            if smin < 1: # 有交集
                if usage_s_range is None:
                    found_linesements = tpts
                    usage_s_range = [smin, smax]
                else:
                    if usage_s_range[1] - usage_s_range[0] < smax - smin:
                        found_linesements = tpts
                        usage_s_range = [smin, smax]
    return found_linesements
    
# 查找匹配的方形角点
def findMatchSquareCorners(usage_square, det_corners, T_corner):
    found_corners = []
    
    # print(usage_square, det_corners)
    
    for each_gt_pt in usage_square:
        dist = np.linalg.norm(det_corners - each_gt_pt, axis=1)
        # print(np.min(dist))
        idx = np.argmin(dist)
        if dist[idx] < T_corner:
            found_corners.append(det_corners[idx])
    found_corners = np.array(found_corners)
    if len(found_corners) != 4:
        found_corners = None
    
    return found_corners

        
    


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
        for it in range(maxIteration):

            # Samples 3 random points
            
            id_samples = random.sample(range(0, n_points), 3)
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
    
    def fit_fixnorm(self, pts, usage_norm, thresh=0.05, minPoints=3, maxIteration=1000):
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
        
        tnorm = np.array([usage_norm])
        
        K = np.matmul(tnorm, pts.transpose())[0, :]
        
        
        
        
        for it in range(maxIteration):

            # Samples 3 random points
            
            id_samples = random.sample(range(0, n_points), minPoints)
            
            K_samples = K[id_samples]
            
            estD = np.mean(K_samples)
            
            pt_id_inliers = []  # list of inliers ids
            dist_pt = np.abs(K - estD)
            
            plane_eq = np.array([usage_norm[0], usage_norm[1], usage_norm[2], -estD])
            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            self.inliers = best_inliers
            self.equation = best_eq
            

        return self.equation, self.inliers

