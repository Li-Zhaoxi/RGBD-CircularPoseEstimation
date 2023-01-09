import bpy
import numpy as np
import time
from scipy.spatial.transform import Rotation 

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens  # lens
    
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height

    
    if camd.sensor_fit == 'AUTO':
        dsize = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_u = dsize
        s_v = dsize
    else:
        assert(0)
        
    '''
    if camd.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    '''

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels
    
    K = np.array([[alpha_u, skew, u_0], [0, alpha_v, v_0], [0, 0, 1]], dtype=np.float)

    return K


def setCamera6DPose(camd, loc:np.ndarray, euler: np.ndarray):
    camd.location[0] = loc[0]
    camd.location[1] = loc[1]
    camd.location[2] = loc[2]
    
    # warning: mode
    camd.rotation_euler[0] = euler[0]
    camd.rotation_euler[1] = euler[1]
    camd.rotation_euler[2] = euler[2]

def motion_sample(dis_step, dis_idxst, dis_idxed, 
                  delta_step, delta_idxst, delta_idxed,
                  theta_step, theta_idxst, theta_idxed):
    sample_dist = [dis_step * idx for idx in range(dis_idxst, dis_idxed + 1)]
    sample_dist = np.array(sample_dist, dtype = np.float)
    
    sample_deltas = [delta_step * idx for idx in range(delta_idxst, delta_idxed + 1)]
    sample_deltas = np.array(sample_deltas, dtype = np.float) / 180.0 * np.pi
    
    sample_theta = [theta_step * idx for idx in range(theta_idxst, theta_idxed + 1)]
    sample_theta = np.array(sample_theta, dtype = np.float) / 180.0 * np.pi
    
    motion_postion = []
    motion_eulerxyz = []
    
    for each_dist in sample_dist:  # 
        for each_delta in sample_deltas:
            for each_theta in sample_theta:
                lxy = each_dist * np.math.sin(each_delta)
                px = lxy * np.math.cos(each_theta)
                py = lxy * np.math.sin(each_theta)
                pz = each_dist * np.math.cos(each_delta)
                
                pos = np.array([px, py, pz], dtype = np.float)
                
                motion_postion.append(pos)
                
                
                
                
                nz = np.array([px, py, pz])
                nx = np.array([-py, px, 0])
                ny = np.array([-px * pz, - py * pz, px * px + py * py])
                if ny[2] < 0:
                    nx *= -1
                    ny *= -1
                nx /= np.linalg.norm(nx)
                ny /= np.linalg.norm(ny)
                nz /= np.linalg.norm(nz)
                
                rotMat = np.vstack([nx, ny, nz]).transpose()
                
                rot = Rotation.from_matrix(rotMat)
                
                euler = rot.as_euler('xyz', degrees=False)
                
                '''
                usage_theta = each_theta + np.pi/2
                if usage_theta >= 2 * np.pi:
                    usage_theta -= 2 * np.pi
                euler = np.array([each_delta, 0, usage_theta])
                '''
                
                motion_eulerxyz.append(euler)
    return motion_postion, motion_eulerxyz
    
def position():
    pass


class CircularPoseSimulator:
    def __init__(self, camera_name = 'Camera', scene_name = 'Scene'):
        self.motion_postion = []
        self.motion_eulerxyz = []
        self.camera_name = camera_name
        self.scene_name = scene_name
        self.idxst = -1
        self.idxed = -1
        
        self.camK = get_calibration_matrix_K_from_blender(
                                    bpy.data.objects[self.camera_name].data)
                                    
        print('camK={0}'.format(self.camK))
                                    
    
    def getMotionSamples(self, dis_step, dis_idxst, dis_idxed, 
                  delta_step, delta_idxst, delta_idxed,
                  theta_step, theta_idxst, theta_idxed):
        self.motion_postion, self.motion_eulerxyz = motion_sample(
                                    dis_step, dis_idxst, dis_idxed, 
                                    delta_step, delta_idxst, delta_idxed,
                                    theta_step, theta_idxst, theta_idxed)
                                    
    def setSceneStartFrame(self, idxst):
        self.idxst = idxst
        bpy.data.scenes[self.scene_name].frame_start = idxst
    
    def setSceneEndFrame(self, idxed):
        self.idxed = idxed
        bpy.data.scenes[self.scene_name].frame_end = idxed
    
    def setSceneCurrentFrame(self, idxnow):
        assert(self.idxst <= idxnow < self.idxed)
        bpy.data.scenes[self.scene_name].frame_current = idxnow
        
    def saveData(self, save_path):
        assert(len(self.motion_postion) == len(self.motion_eulerxyz))
    
        
        np.savez(save_path, motion_postion=self.motion_postion, 
                    motion_eulerxyz=self.motion_eulerxyz,
                    camK = self.camK)
        
    
    def animation(self):
        assert(len(self.motion_postion) == len(self.motion_eulerxyz))
        num_pose = len(self.motion_postion)
        
        self.setSceneStartFrame(0)
        self.setSceneEndFrame(num_pose)
        
        
        camera_object = bpy.data.objects[self.camera_name]
        idx_frame_now = 0
        for idx_pose in range(num_pose):
#            idx_pose = 1
            bpy.ops.render.render(animation=False) # Use these can generate usage images
            print(idx_pose, self.motion_postion[idx_pose], self.motion_eulerxyz[idx_pose])
            self.setSceneCurrentFrame(idx_frame_now)
            idx_frame_now += 1
            setCamera6DPose(camera_object, self.motion_postion[idx_pose], self.motion_eulerxyz[idx_pose])
#            break
#            bpy.context.view_layer.update()
#            time.sleep(1)

            
        
    
        
def main():
    
    camera_name = 'Camera'
    
    # # [1.5 - 20]m
    # dis_step = 0.5
    # dis_idxst = 6
    # dis_idxed = 30
    
    # # [5, 80]^o
    # delta_step = 2.5
    # delta_idxst = 1
    # delta_idxed = 32
    
    # # [0, 330]^o
    # theta_step = 15
    # theta_idxst = 0
    # theta_idxed = 22
    
    # [1.5 - 20]m
    dis_step = 1
    dis_idxst = 6
    dis_idxed = 20
    
    # [5, 80]^o
    delta_step = 5
    delta_idxst = 1
    delta_idxed = 16
    
    # [0, 330]^o
    theta_step = 30
    theta_idxst = 0
    theta_idxed = 11
    
    pose_sim = CircularPoseSimulator(camera_name = camera_name)
    
    pose_sim.getMotionSamples(dis_step, dis_idxst, dis_idxed, delta_step, delta_idxst, delta_idxed, theta_step, theta_idxst, theta_idxed)
#    pose_sim.animation()
    pose_sim.saveData('C:/Users/Zhaoxi-Li/Desktop/PoseSimulator/data/data')
    
    print('finish')


    

if __name__ == '__main__':
    main()