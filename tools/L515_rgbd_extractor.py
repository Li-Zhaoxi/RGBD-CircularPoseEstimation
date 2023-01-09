import pyrealsense2 as rs
import numpy as np
import cv2
from realsense_expand import extract_color_profile, extract_depth_profile, get_depth_scale
import os
from PIL import Image
# stream_profile

need_color_image = True  # 是否需要彩色图
need_depth_image = True  # 是否需要深度图
need_rgbd_align = True   # 是否需要对齐

need_intrinsics = True   # 是否需要内参

need_visualize = False    # 是否需要可视化RGBorD

need_read_bags = True    # 从bag中读取文件
bag_file_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark3/20220125_004616.bag'


need_save_rgbd = False
save_color_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark3/color/'
save_depth_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark3/depth/'

pipeline = rs.pipeline()


config = rs.config()

if need_read_bags:
    if need_color_image:
        config.enable_stream(rs.stream.color)
    if need_depth_image:
        config.enable_stream(rs.stream.depth)
    config.enable_device_from_file(bag_file_path);

# Create colorizer object
colorizer = rs.colorizer()

# if need_color_image:
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# if need_depth_image:
#     config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
if need_rgbd_align and need_color_image and need_depth_image:
    # 设定需要对齐的方式（这里是深度对齐彩色，彩色图不变，深度图变换）
    align_to = rs.stream.color
    alignedFs = rs.align(align_to)
else:
    need_rgbd_align = False
    
# if need_read_bags:
#     config.enable_device_from_file(bag_file_path);




profile = pipeline.start(config)
profile.get_device().as_playback().set_real_time(False)

image_count = 0
frame_first_number = -1
while True:
    # 获取图片帧
    frameset = pipeline.wait_for_frames()

    # 输出相机内参信息
    if need_intrinsics:
        if need_color_image:
            color_frame = frameset.get_color_frame()
            color_profile = color_frame.get_profile()
            color_intrinsics = extract_color_profile(color_profile)
            print(color_intrinsics)

        if need_depth_image:
            depth_frame = frameset.get_depth_frame()
            depth_profile = depth_frame.get_profile()
            depth_intrinsics = extract_depth_profile(depth_profile)
            depth_scale = get_depth_scale(profile)
            print(depth_intrinsics)
            print(depth_scale)

        need_intrinsics = False
        # print(color_intrinsics)
        # color_intrinsics.pop('stream_type')
        # print(color_intrinsics)
        # depth_intrinsics.pop('stream_type')
        np.savez('intrinsics', color_intrinsics = color_intrinsics, depth_intrinsics = depth_intrinsics, depth_scale = depth_scale)
        exit()
    # 获取图像信息
    if need_rgbd_align:
        frameset = alignedFs.process(frameset)
        
    image_count += 1
    if need_color_image:
        color_frame = frameset.get_color_frame()
        if not color_frame:
            continue
        current_frame_number = color_frame.get_frame_number()
        # print(frame_first_number, current_frame_number)
        if frame_first_number < 0:
            frame_first_number = current_frame_number
        elif frame_first_number >= current_frame_number and image_count >= 3:
            break
        # frame_number = color_frame.get_frame_number()
        color_img = np.asanyarray(color_frame.get_data())
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        if need_visualize:
            cv2.imshow('color', color_img)
        
        if need_save_rgbd:
            save_path = os.path.join(save_color_root_path, str(image_count).zfill(8) + '.png')
            cv2.imwrite(save_path, color_img)
            print('save color name: ', save_path, frame_first_number, current_frame_number)

    if need_depth_image:
        depth_frame = frameset.get_depth_frame()
        if not depth_frame:
            continue
        frame_number = depth_frame.get_frame_number()
        depth_img = np.asanyarray(depth_frame.get_data())
        # depth_bgr = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.003), cv2.COLORMAP_JET)
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        depth_color_image = cv2.cvtColor(depth_color_image, cv2.COLOR_RGB2BGR)

        
        
        # print(np.min(depth_img), np.max(depth_img))
        # print(np.min(depth_color_image), np.max(depth_color_image), depth_color_image.dtype)

        if need_visualize:
            cv2.imshow('depth', depth_color_image)
            
        if need_save_rgbd:
            save_path = os.path.join(save_depth_root_path, str(image_count).zfill(8) + '.png')
            cv2.imwrite(save_path, depth_img)
            save_path = os.path.join(save_depth_root_path, str(image_count).zfill(8) + '.jpg')
            cv2.imwrite(save_path, depth_color_image)
    
    # k = cv2.waitKey(0)
    
    # print(frame_first_number)
    # exit()
    
    # if need_read_bags:
    #     if image_count > frame_first_number:
    #         break

    # if k == ord('q'):
    #     break
    # elif k == ord('s'):
    #     # img_profix = os.path.join('D:/img2', ' ' , str(i))
    #     # img_profix = os.path.join('D:/img2/depth', 'depth_', str(i))
    #     img_profix = (str(i))
    #     cv2.imwrite(img_profix + '_color.jpg', color_img)
    #     cv2.imwrite(img_profix + '_depth.png', depth_img)
    #     # cv2.imwrite(img_profix_color + '_color.jpg', color_img)
    #     # cv2.imwrite(img_profix_depth + '_depth.png', depth_img)
    #     i = i + 1
    #     print('saved')
    #     # "D:/img/color", 'color_' + str(i) + '.jpg')