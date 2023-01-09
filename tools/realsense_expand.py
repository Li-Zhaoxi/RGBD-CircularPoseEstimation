import pyrealsense2 as rs2
from ElpPy.utils import GeneralCamera
import numpy as np


# 获得两个设备的外参
def get_extrinsics(profile_src, profile_dst):
    extrin = profile_src.get_extrinsics_to(profile_dst)
    return extrin

# 获得深度scale
def get_depth_scale(profile):
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    return depth_scale

# 获得彩色图内参
def extract_color_profile(color_profile):
    fps = color_profile.fps()  # 30
    stream_index = color_profile.stream_index()  # 0
    stream_name = color_profile.stream_name()  # Depth
    # stream_type = color_profile.stream_type()  # stream.\
        
    # rs2.pyrealsense2.intrinsics
        
    unique_id = color_profile.unique_id()
    cvsprofile = rs2.video_stream_profile(color_profile)
    intrinsics = cvsprofile.get_intrinsics()
    
    print(intrinsics)
    print(intrinsics.coeffs, type(intrinsics.coeffs))
    print(str(intrinsics.model))
    gcam = GeneralCamera(intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy, 
                         dist_model = str(intrinsics.model), coeffs=np.array(intrinsics.coeffs))
    print(gcam.dist_model)
    isize = np.array([intrinsics.width, intrinsics.height])
    

    return {'fps': fps,
            'stream_index': stream_index,
            'stream_name': stream_name,
            'unique_id': unique_id,
            'camera': gcam,
            'size': isize}


# 获得深度图内参
def extract_depth_profile(depth_profile):
    fps = depth_profile.fps()  # 30
    stream_index = depth_profile.stream_index()  # 0
    stream_name = depth_profile.stream_name()  # Depth
    # stream_type = depth_profile.stream_type()  # stream.\
    unique_id = depth_profile.unique_id()
    dvsprofile = rs2.video_stream_profile(depth_profile)

    # width: 640, height: 480, ppx: 317.78, ppy: 236.709, fx: 382.544, fy: 382.544, model: 4, coeffs: [0, 0, 0, 0, 0]
    intrinsics = dvsprofile.get_intrinsics()
    gcam = GeneralCamera(intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy, 
                         dist_model = str(intrinsics.model), coeffs=np.array(intrinsics.coeffs))
    isize = np.array([intrinsics.width, intrinsics.height])
    
    dist_model = str(intrinsics.model)
    coeffs = np.array(intrinsics.coeffs)
    
    return {'fps': fps,
            'stream_index': stream_index,
            'stream_name': stream_name,
            # 'stream_type': stream_type,
            'unique_id': unique_id,
            'camera': gcam,
            'size': isize}
