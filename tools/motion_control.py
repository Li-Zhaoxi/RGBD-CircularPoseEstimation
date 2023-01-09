

from math import degrees
import numpy as np
import logging
from multiprocessing import Process, Queue
import time
from AUBOControler.robotcontrol import Auboi5Robot, RobotError, RobotErrorType, logger_init, runWaypoint
from scipy.spatial.transform import Rotation
logger = logging.getLogger('main.robotcontrol')


# void Robot::Move_rotX()
# {
# 	double rad = 10.0 / 180 * CV_PI;
# 	Vec6d current_pos = m_robotPose;
# 	Eigen::Vector3d eulerAngle(current_pos[3] / 180 * F_PI, current_pos[4] / 180 * F_PI, current_pos[5] / 180 * F_PI);
# 	Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle(0), Eigen::Vector3d::UnitX()));
# 	Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle(1), Eigen::Vector3d::UnitY()));
# 	Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle(2), Eigen::Vector3d::UnitZ()));
# 	Eigen::AngleAxisd rotation_vector_1;
# 	rotation_vector_1 = yawAngle * pitchAngle * rollAngle; // 列为表示的向量
# 	//rotation_vector_1 = rollAngle * pitchAngle * yawAngle; // 列为表示的向量
# 	Eigen::AngleAxisd rotation_vector(rad, Eigen::Vector3d(1, 0, 0));
# 	Eigen::Matrix3d rotation_matrix3d = rotation_vector_1.matrix() * rotation_vector.matrix();
# 	Eigen::Vector3d euler_angles = rotation_matrix3d.eulerAngles(2, 1, 0);
# 	current_pos[3] = euler_angles(2) / F_PI * 180;
# 	current_pos[4] = euler_angles(1) / F_PI * 180;
# 	current_pos[5] = euler_angles(0) / F_PI * 180;
# 	m_robotPose = current_pos;
# 	SCR_5_MoveJ(current_pos, speed, acc, 0, 0);

 
def move_rotX(current_quat, rot_angle):
    rad =rot_angle / 180.0 * np.pi
    
    cvt_pose = Rotation.from_quat(current_quat)
    rotation_vector_1 = cvt_pose.as_matrix()
    
    rot_rad = Rotation.from_rotvec(rad * np.array([0, 0, 1]))
    
    rotation_matrix3d = np.matmul(rot_rad.as_matrix(), rotation_vector_1)
    # rotation_matrix3d = np.matmul(rotation_vector_1, rot_rad.as_matrix())

    return Rotation.from_matrix(rotation_matrix3d).as_quat()

    


def pcd_control_test():
    joint_radian = (83.213684 / 180.0 * np.pi, -20.109660 / 180.0 * np.pi, -67.889615 / 180.0 * np.pi, 
                                -49.398489 / 180.0 * np.pi, -84.059548 / 180.0 * np.pi, 0.1 / 180.0 * np.pi)
    print(joint_radian)
    joint_radian = move_rotX(joint_radian, 30)
    
    print(joint_radian)

def pcd_control_strategy():
    
    mode_record_current_joint_angles = False
    mode_control_path_strategy_PCD = True
    # 初始化logger
    logger_init()
    # 启动测试
    logger.info("{0} test beginning...".format(Auboi5Robot.get_local_time()))
    
    # 系统初始化
    Auboi5Robot.initialize()
    # 创建机械臂控制类
    robot = Auboi5Robot()
    # 创建上下文
    handle = robot.create_context()
    # 打印上下文
    logger.info("robot.rshd={0}".format(handle))
    
    try:
        # queue = Queue()
        # p = Process(target=runWaypoint, args=(queue,))
        # p.start()
        # time.sleep(5)
        # print("process started.")
        
        ip = '172.17.140.192'
        port = 8899
        result = robot.connect(ip, port)
        
        if result != RobotErrorType.RobotError_SUCC:
            logger.info("connect server{0}:{1} failed.".format(ip, port))
        else:
            robot.project_startup()
            robot.enable_robot_event()
            robot.init_profile()
            joint_maxvelc = (2.596177 / 4, 2.596177 / 4, 2.596177 / 4, 3.110177 / 4, 3.110177 / 4, 3.110177 / 4)
            joint_maxacc = (17.308779/2.5, 17.308779/2.5, 17.308779/2.5, 17.308779/2.5, 17.308779/2.5, 17.308779/2.5)
            robot.set_joint_maxacc(joint_maxacc)
            robot.set_joint_maxvelc(joint_maxvelc)
            # robot.set_arrival_ahead_blend(0.05)
        
        # 记录起始点模式
        if mode_record_current_joint_angles:
            current_joints_info = robot.get_current_waypoint()
            np.savez('current_joints', joint = current_joints_info['joint'], 
                     pos = current_joints_info['pos'],
                     ori = current_joints_info['ori'])
            print('finish mode_record_current_joint_angles')
        
        if mode_control_path_strategy_PCD:
            start_joint = np.load('current_joints.npz', allow_pickle=True)
            joint_radian_start = start_joint['joint']
            for idx_time in range(5):
                # time.sleep(1)
                print(joint_radian_start)
                print(np.array(joint_radian_start)/np.pi*180.0)
                robot.move_joint(joint_radian_start.tolist(), True)
                
                joint_radian = joint_radian_start
                robot_pose_so3 = robot.forward_kin(joint_radian)
                next_quat = move_rotX(robot_pose_so3['ori'], -10 * (idx_time + 1))
                
                combo_radian = robot.inverse_kin(joint_radian, robot_pose_so3['pos'], next_quat)
                joint_radian = combo_radian['joint']
                print(joint_radian)
                print(np.array(joint_radian)/np.pi*180.0)
                robot.move_joint(joint_radian, True)
                
                
                
                
                # joint_radian = move_rotX(joint_radian, 30)
                # robot.move_joint(joint_radian, True)
                
                # joint_radian = (55.5/180.0*np.pi, -20.5/180.0*np.pi, -72.5/180.0*np.pi, 
                #                 38.5/180.0*np.pi, -90.5/180.0*np.pi, 55.5/180.0*np.pi)
                # robot.move_joint(joint_radian, True)
                
                print("-----------------------------")

                # queue.put(joint_radian)

                robot.project_stop()
        
        robot.disconnect()
        
    except KeyboardInterrupt:
        robot.move_stop()
    except RobotError as e:
        logger.error("robot Event:{0}".format(e))
        
    finally:
        # 断开服务器链接
        if robot.connected:
            # 断开机械臂链接
            robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()
        print("run end-------------------------")
        
if __name__ == '__main__':
    # test_process_demo()
    pcd_control_strategy()
    # pcd_control_test()
    logger.info("test completed")