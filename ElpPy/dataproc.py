import cv2
import numpy as np
import json
import os
from scipy.io import loadmat
from ElpPy.utils import GeneralCamera, GeneralLineSegment, GeneralSpacialCircle, drawCirclePose
from PoseLabel.Shapes import Shapes
from PoseLabel.label_utils import cvtQPtfs2pts 

# pixel value range: [0-255]
def imnoise_gauss(img, standard_gauss_mask, mu = 0.0, var = 0.01):
    assert(np.all(img.shape == standard_gauss_mask.shape))
    
    imgnoise = img.astype(np.float) / 255 + np.sqrt(var)*standard_gauss_mask + mu;
    
    imgnoise = np.round(imgnoise * 255).astype('uint8')
    imgnoise = np.clip(imgnoise, 0, 255)
    
    return imgnoise

def filterDisp(disp, dot_pattern_, invalid_disp_):
    
    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp

def add_gaussian_shifts(depth, gaussian_shift_x, gaussian_shift_y, std=1/2.0):
    rows, cols = depth.shape 
    
    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)
    
    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)
    
    xp = xp.astype(np.float)
    yp = yp.astype(np.float)
    
    xp_interp = np.minimum(np.maximum(xp + gaussian_shift_x * std, 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shift_y * std, 0.0), rows)
    
    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp








class GTPoseLoader(object):
    def __init__(self) -> None:
        super().__init__()
        
        self.is_valid = False
        
        self.circular_pose_15plane = 'CircularPose-15PlaneBlender'
        self.circular_pose_realring40traj = 'CircularPose-RealRing40Traj'
        self.circular_gt_only = 'CircularPose-GTOnly' # 仅有标记的结果，RGB+GT+Mask
        
        self.all_dataset_name = {self.circular_pose_15plane, 
                                 self.circular_pose_realring40traj,
                                 self.circular_gt_only}
        self.usage_ds_name = None
        self.all_dataset_gt = None
        self.total_data = None
        self.root_path = None
        
    
    def __len__(self):
        if self.total_data is not None:
            return self.total_data
        else:
            return 0;
        
    def load(self, dataset_root_path, dataset_name):
        assert(dataset_name in self.all_dataset_name)
        assert(os.path.exists(dataset_root_path))
        
        if dataset_name == self.circular_pose_15plane:
            json_path = os.path.join(dataset_root_path, 'data.json')
            with open(json_path, 'r') as load_f:
                gt_dict = json.load(load_f)
                self.usage_ds_name = dataset_name
                self.all_dataset_gt = gt_dict
                # print(list(gt_dict.keys()))
                self.root_path = dataset_root_path
                
            self.total_data = len(self.all_dataset_gt)
            
        elif dataset_name == self.circular_pose_realring40traj:
            self.root_path = dataset_root_path
            self.usage_ds_name = dataset_name
            
            # gt.txt存的是json图像名
            # traj.mat 存的是相邻两个轨迹之间的位姿变换
            img_name_path = os.path.join(dataset_root_path, 'gt.txt')
            traj_name_path = os.path.join(dataset_root_path, 'traj.mat')
            intrin_path = os.path.join(dataset_root_path, 'intrinsics.npz')
            img_names = [] # 图像名
            with open(img_name_path, 'r') as f:
                for each_line in f.readlines():
                    if len(each_line) < 3:
                        continue
                    tmp_name = each_line.strip('\n')
                    img_names.append(tmp_name)
            
            traj_data = loadmat(traj_name_path)
            end_poses = traj_data['end_pose']
            joint_angles = traj_data['joint_angle']
            
            
            intrin_data = np.load(intrin_path, allow_pickle=True)
            
            color_intrinsics = intrin_data['color_intrinsics'][()]
            gcam = color_intrinsics['camera']
            # print(gcam.dist_model)
            # exit()
            img_size = color_intrinsics['size']
            depth_scale = intrin_data['depth_scale']
            # dist_model = intrin_data['dist_model']
            # coeffs = intrin_data['coeffs']
            
            shapeinfo_path = os.path.join(dataset_root_path, 'shapeinfo.mat')
            shape_info = loadmat(shapeinfo_path)
            
            
            self.all_dataset_gt = {}
            self.all_dataset_gt['size'] = img_size
            self.all_dataset_gt['gcam'] = gcam
            self.all_dataset_gt['depth_scale'] = depth_scale
            
            self.all_dataset_gt['radius'] = shape_info['radius'][0][0]
            self.all_dataset_gt['square'] = shape_info['square']
            self.all_dataset_gt['line'] = shape_info['line']
            self.all_dataset_gt['TBC'] = shape_info['TBC']
            
            for traj_idx in range(len(img_names)):
                gt_dict = {}
                
                color_suffix = 'color/{0}'.format(img_names[traj_idx])
                depth_suffix = 'depth/{0}'.format(img_names[traj_idx].split('.')[0] + '.png')
                
                gt_dict['imgname'] = img_names[traj_idx]
                gt_dict['color_full_path'] = os.path.join(dataset_root_path, 'color', img_names[traj_idx])
                gt_dict['gt_full_path'] = os.path.join(dataset_root_path, 'gt', img_names[traj_idx] + '.json')
                gt_dict['mask_full_path'] = os.path.join(dataset_root_path, 'mask', img_names[traj_idx] + '.png')
                gt_dict['depth_full_path'] = os.path.join(dataset_root_path, 'depth', img_names[traj_idx].split('.')[0] + '.png')
                
                
                shape_json_path = gt_dict['gt_full_path']
                gt_shape = Shapes()
                gt_shape.loadJson(shape_json_path)
                
                # 处理标记出的椭圆
                elps = [t['gelp'] for t in gt_shape.labeled_elps]
                gt_dict['gtellipse'] = elps[0] if elps[0].getArea() > elps[1].getArea() else elps[1]
    
                # 处理标记出的直线
                pts = cvtQPtfs2pts(gt_shape.labeled_lines[0]['pts'])
                cor_line = GeneralLineSegment(pts[0], pts[1])
                # cor_line = gt_shape.labeled_lines[0]['gline']
                # print(gt_shape.labeled_lines[0]['pts'])
                # exit()
                # GeneralLineSegment
                
                # self.labeled_lines.append({'pts': self.selected_line_pts,
                #                       'gline': self.selected_line_shape})
                
                gt_dict['leftline'] = cor_line
                
                # 处理标记出的点
                gt_square = gt_shape.labeled_pts
                gt_dict['innersquare'] = gt_square
                
                
                # print(joint_angles[0])
                gt_dict['end_pose'] = np.array(end_poses[0][traj_idx])
                gt_dict['joint_angle'] = np.array(joint_angles[0][traj_idx])
                
                gt_dict['color_suffix'] = color_suffix
                gt_dict['depth_suffix'] = depth_suffix
                
                self.all_dataset_gt[traj_idx] = gt_dict
                
            self.total_data = len(img_names)
            
            # print(traj_data)
            # print(len(img_names))
            
        elif dataset_name == self.circular_gt_only:
            self.root_path = dataset_root_path
            self.usage_ds_name = dataset_name
            
            # gt.txt存的是json图像名
            # traj.mat 存的是相邻两个轨迹之间的位姿变换
            img_name_path = os.path.join(dataset_root_path, 'gt.txt')
            
            img_names = [] # 图像名
            with open(img_name_path, 'r') as f:
                for each_line in f.readlines():
                    if len(each_line) < 3:
                        continue
                    tmp_name = each_line.strip('\n')
                    img_names.append(tmp_name)
            
            
            
            self.all_dataset_gt = {}
            for traj_idx in range(len(img_names)):
                gt_dict = {}
                gt_dict['imgname'] = img_names[traj_idx]
                gt_dict['color_full_path'] = os.path.join(dataset_root_path, 'color', img_names[traj_idx])
                gt_dict['gt_full_path'] = os.path.join(dataset_root_path, 'gt', img_names[traj_idx] + '.json')
                gt_dict['mask_full_path'] = os.path.join(dataset_root_path, 'mask', img_names[traj_idx] + '.png')
                
                self.all_dataset_gt[traj_idx] = gt_dict
                
            self.total_data = len(img_names)
            
        else:
            assert(0)
    
    # 返回值:
    # Spacial Circular
    
    def __getitem__(self, index, read_rgbd = False):
        # print('index', index)
        assert(0 <= index < self.total_data)
        
        res_gt = {}
        
        if self.usage_ds_name == self.circular_pose_15plane:
            # print(list(self.all_dataset_gt.keys()))
            usage_gt = self.all_dataset_gt[str(index)]
            
            usage_gscir = []
            usage_innersquare = []
            usage_innersquare_W = []
            usage_leftlines = []
            usage_leftlines_W = []
            usage_full_Rot = []
            
            for each_gt in usage_gt['spacialcircle']:
                gscir = GeneralSpacialCircle.returnOne(each_gt)
                usage_gscir.append(gscir)
            
            for each_gt in usage_gt['innersquare']:
                usage_innersquare.append(np.array(each_gt))
            
            # print('usage_gt[innersquare_W]', usage_gt['innersquare_W'])
            for each_gt in usage_gt['innersquare_W']:
                # print(each_gt)
                usage_innersquare_W.append(np.array(each_gt).transpose())
                
            for each_gt in usage_gt['leftline']:
                usage_leftlines.append(np.array(each_gt))
            
            for each_gt in usage_gt['leftline_W']:
                usage_leftlines_W.append(np.array(each_gt))
                
            for each_Rot in usage_gt['Rot']:
                usage_full_Rot.append(np.array(each_Rot))

            res_gt['spacialcircle'] = usage_gscir
            res_gt['innersquare'] = usage_innersquare
            res_gt['innersquare_W'] = usage_innersquare_W
            res_gt['leftline'] = usage_leftlines
            res_gt['leftline_W'] = usage_leftlines_W
            res_gt['full_rot'] = usage_full_Rot
            
            
            res_gt['camera'] = GeneralCamera.returnOne(usage_gt['camera'])
            res_gt['size'] = np.array(usage_gt['size'])
            res_gt['color_suffix'] = usage_gt['color_suffix']
            res_gt['depth_suffix'] = usage_gt['depth_suffix']
            res_gt['root_path'] = self.root_path
            
            res_gt['color_full_path'] = os.path.join(self.root_path, res_gt['color_suffix'])
            res_gt['depth_full_path'] = os.path.join(self.root_path, res_gt['depth_suffix'])
            
            if read_rgbd:
                color_full_path = res_gt['color_full_path']
                imgC = cv2.imread(color_full_path)
                imgG = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
                depth_full_path = res_gt['depth_full_path']
                imgD = cv2.imread(depth_full_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
                imgD = imgD.astype(np.float)
                imgD[imgD > 200] = 0
                res_gt['imgC'] = imgC
                res_gt['imgG'] = imgG
                res_gt['imgD'] = imgD
                # print(imgD.shape, imgG.shape, res_gt['size'])
                assert(np.all(imgD.shape == imgG.shape) and np.all(imgG.shape == res_gt['size'][[1,0]]))
        elif self.usage_ds_name == self.circular_pose_realring40traj:
            usage_gt = self.all_dataset_gt[index]
            
            res_gt['color_full_path'] = usage_gt['color_full_path']
            res_gt['depth_full_path'] = usage_gt['depth_full_path']
            res_gt['mask_full_path'] = usage_gt['mask_full_path']
            res_gt['color_suffix'] = usage_gt['color_suffix']
            res_gt['depth_suffix'] = usage_gt['depth_suffix']
            res_gt['root_path'] = self.root_path
            
            
            res_gt['gtellipse'] = usage_gt['gtellipse']
            res_gt['innersquare'] = cvtQPtfs2pts(usage_gt['innersquare'])  # 有序4个点
            res_gt['leftline'] = usage_gt['leftline']  # 直线段端点
            
            
            res_gt['size'] = self.all_dataset_gt['size']
            res_gt['camera'] = self.all_dataset_gt['gcam']
            
            # gcam = self.all_dataset_gt['gcam']
            # print(gcam.dist_model)
            # exit()
            depth_scale = self.all_dataset_gt['depth_scale']
            res_gt['depth_scale'] = depth_scale
            
            res_gt['radius'] = self.all_dataset_gt['radius']
            res_gt['innersquare_W'] = self.all_dataset_gt['square']
            res_gt['leftline_W'] = self.all_dataset_gt['line']
            res_gt['TBC'] = self.all_dataset_gt['TBC']
            
            
            res_gt['end_pose'] = usage_gt['end_pose']
            res_gt['joint_angle'] = usage_gt['joint_angle']

            
            
            if read_rgbd:
                color_full_path = res_gt['color_full_path']
                imgC = cv2.imread(color_full_path)
                imgG = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
                depth_full_path = res_gt['depth_full_path']
                imgD = cv2.imread(depth_full_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                imgD = imgD.astype(np.float) * depth_scale
                res_gt['imgC'] = imgC
                res_gt['imgG'] = imgG
                res_gt['imgD'] = imgD
                # print(imgD.shape, imgG.shape, res_gt['size'])
                # print(res_gt['size'][[1,0]])
                # exit()
                assert(np.all(imgD.shape == imgG.shape) and np.all(imgG.shape == res_gt['size'][[1,0]]))
        elif self.usage_ds_name == self.circular_gt_only:
            usage_gt = self.all_dataset_gt[index]
            
            res_gt['color_full_path'] = usage_gt['color_full_path']
            res_gt['mask_full_path'] = usage_gt['mask_full_path']
            res_gt['size'] = [1280, 720]
            if read_rgbd:
                color_full_path = res_gt['color_full_path']
                imgC = cv2.imread(color_full_path)
                imgG = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
                res_gt['imgC'] = imgC
                # print(imgD.shape, imgG.shape, res_gt['size'])
                assert(np.all(imgG.shape == res_gt['size'][[1,0]]))
        else:
            assert(0)
                
        return res_gt
    
    
    def drawGT(self, index):
        
        usage_gt = self.__getitem__(index, read_rgbd=True)
        
        
        if self.usage_ds_name == self.circular_pose_15plane:
            imgT = np.copy(usage_gt['imgC'])
            for each_gscir, each_insquare, each_line in zip(usage_gt['spacialcircle'], usage_gt['innersquare'], usage_gt['leftline']):
                drawCirclePose(imgT, usage_gt['camera'], each_gscir, square_pixels=each_insquare, cooperate_line=each_line)
            return imgT
        if self.usage_ds_name == self.circular_pose_realring40traj:
            imgT = np.copy(usage_gt['imgC'])
            
            # 绘制椭圆
            usage_elp = usage_gt['gtellipse']
            usage_elp.drawEllipse(imgT, (0, 0, 255), 2)
            
            # 绘制直线
            usage_line = usage_gt['leftline']
            line_pts = usage_line.pts
            # print(line_pts)
            assert(len(line_pts) == 2)
            cv2.line(imgT, (int(line_pts[0][0] + 0.5), int(line_pts[0][1] + 0.5)),
                        (int(line_pts[1][0] + 0.5), int(line_pts[1][1] + 0.5)), (0, 255, 0), 2)
            
            
            square_pixels = usage_gt['innersquare']
            if square_pixels is not None and len(square_pixels) > 0:
                for each_pts in square_pixels:
                # print(each_pts)
                    cv2.circle(imgT, (int(each_pts[0] + 0.5), int(each_pts[1] + 0.5)), 3, (0 , 255, 0), 2)
                # print(square_pixels)
                cv2.line(imgT, (int(square_pixels[0, 0] + 0.5), int(square_pixels[0, 1] + 0.5)), 
                    (int(square_pixels[1, 0] + 0.5), int(square_pixels[1, 1] + 0.5)), (0, 0, 255), 2)
            
            return imgT
            
        else:
            assert(0)
    
    
    