import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

align_to = rs.stream.color
alignedFs = rs.align(align_to)
profile = pipeline.start(config)

frames = pipeline.wait_for_frames()
depth = frames.get_depth_frame()
color = frames.get_color_frame()
# 获取内参
depth_profile = depth.get_profile()
print(depth_profile)
# <pyrealsense2.video_stream_profile: 1(0) 640x480 @ 30fps 1>
print(type(depth_profile))
# <class 'pyrealsense2.pyrealsense2.stream_profile'>
print(depth_profile.fps())
# 30
print(depth_profile.stream_index())
# 0
print(depth_profile.stream_name())
# Depth
print(depth_profile.stream_type())
# stream.depth
print('', depth_profile.unique_id)
# <bound method PyCapsule.unique_id of <pyrealsense2.video_stream_profile: 1(0) 640x480 @ 30fps 1>>

color_profile = color.get_profile()

print(depth_profile.fps())
# 30
print(depth_profile.stream_index())
# 0

cvsprofile = rs.video_stream_profile(color_profile)
dvsprofile = rs.video_stream_profile(depth_profile)

color_intrin = cvsprofile.get_intrinsics()
print(color_intrin)
# width: 640, height: 480, ppx: 318.482, ppy: 241.167, fx: 616.591, fy: 616.765, model: 2, coeffs: [0, 0, 0, 0, 0]
depth_intrin = dvsprofile.get_intrinsics()
print(depth_intrin)
# width: 640, height: 480, ppx: 317.78, ppy: 236.709, fx: 382.544, fy: 382.544, model: 4, coeffs: [0, 0, 0, 0, 0]
extrin = depth_profile.get_extrinsics_to(color_profile)
print(extrin)
# rotation: [0.999984, -0.00420567, -0.00380472, 0.00420863, 0.999991, 0.00076919, 0.00380145, -0.00078519, 0.999992]
# translation: [0.0147755, 0.000203265, 0.00051274]
# 设定需要对齐的方式（这里是深度对齐彩色，彩色图不变，深度图变换）



depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print('depth scale: ', depth_scale)

while True:
    # 获取图片帧
    frameset = pipeline.wait_for_frames()
    aligned_frames = alignedFs.process(frameset)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # depth_frame = frameset.get_depth_frame()
    # color_frame = frameset.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    depth_img = np.asanyarray(depth_frame.get_data())
    color_img = np.asanyarray(color_frame.get_data())

    # print(depth_img.dtype)

    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    depth_bgr = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.003), cv2.COLORMAP_JET)
    # cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

    cv2.imshow('test', color_img)
    cv2.imshow('depth', depth_bgr)
    k = cv2.waitKey(10)

    if k == ord('q'):
        break
    elif k == ord('s'):
        # img_profix = os.path.join('D:/img2', ' ' , str(i))
        # img_profix = os.path.join('D:/img2/depth', 'depth_', str(i))
        img_profix = (str(i))
        cv2.imwrite(img_profix + '_color.jpg', color_img)
        cv2.imwrite(img_profix + '_depth.png', depth_img)
        # cv2.imwrite(img_profix_color + '_color.jpg', color_img)
        # cv2.imwrite(img_profix_depth + '_depth.png', depth_img)
        i = i + 1
        print('saved')
        # "D:/img/color", 'color_' + str(i) + '.jpg')