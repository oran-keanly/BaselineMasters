#!/usr/bin/env python3
from audioop import mul
import string
from time import sleep, time
#from unittest import result
import rospy
#import rosbag 
#import bagpy
import csv

from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped,Polygon, Point32, PoseWithCovarianceStamped, PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np

import copy
from agent_utils import get_actuation, nearest_point_on_trajectory_py2, first_point_on_trajectory_intersecting_circle, quaternion_to_euler_angle

from mapping import map
import functions as func
from optimization import Results
from optimization import Kinematic_Bicycle_Model_States
from track import Track
from parameters import Kinematic_Bicycle_Model_Parameters
from optimization import Optimizer
#from scipy.integrate import odeint

class Agent(object):
    def __init__(self):
        # TODO: load waypoints from csv
        #self.trajectory = Trajectory('columbia_small')
        #self.waypoints = self.trajectory.waypoints
        #self.safe_speed = 0.5
        pass


class My_Agent(Agent):
    def __init__(self, map_name):
        super(My_Agent, self).__init__()
        #results = self.get_plan_from_csv(r"./Reference for columbia_small/Optimization of columbia_small/results.csv")

        self.have_plan = False
        
        m = map(map_name)
        self.global_track = m.generate_track(0.1)
        single_track = self.global_track
        single_lap_N = len(single_track.curvature)
        track_twice = np.zeros(single_lap_N*2)
        two_lap_N = single_lap_N*2
        
        for i in range(single_lap_N):
            track_twice[i] = single_track.curvature[i]
        for i in range(single_lap_N, two_lap_N):
            track_twice[i] = single_track.curvature[i-single_lap_N]
        two_lap_track = Track(two_lap_N, 0.1, track_twice, 1.4)


        self.laps = 0
            
        self.position = np.array([(0 - self.global_track.origin[0]), (0 - self.global_track.origin[1])])
        self.theta = 0
        self.orientation_x = self.global_track.yaw_angle[0] - 0.2
        self.previous_s = 0
        self.previous_index = 0
        
        self.drive = None 
        self.prev_pf_odom_msg = None

        self.long_vel = 0
        self.lat_vel = 0
        self.r_z = 0
       
        self.propagated_x = 0
        self.propagated_y = 0
        self.propagated_theta = 0
        self.first_scan_callback = True
        self.test = 0
        self.Stanley_Gain = 1.0

        self.drive_pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback, 10) 
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.pf_odom_subscriber = rospy.Subscriber('/pf/pose/odom', Odometry, self.pf_odom_callback, 10)
        self.propagated_odom_pub = rospy.Publisher('/propagated_odom', Odometry, queue_size=1)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        
        self.init_conditions = Kinematic_Bicycle_Model_States(0, 0, 0, 0, 1.0, 0, 0, 0, 0)
        self.model_parameters = Kinematic_Bicycle_Model_Parameters(Slip_AngleReg=5, Steer_Reg=5, Max_Tyre_Force=15.0, Motor_gain=8, Friction_Ellipse=2, Drag_Resist=3.0)
        self.opt = Optimizer(self.global_track, self.model_parameters, print_level=0)
        results = self.opt.optimize(self.init_conditions, self.model_parameters, Temp_Track = two_lap_track)#, export_results=True, save_path="./test/results1")
        
        first_lap_delta_st = results.delta_st[0:single_lap_N]
        first_lap_delta_T = results.delta_T[0:single_lap_N]
        first_lap_n = results.n[0:single_lap_N]
        first_lap_mu = results.mu[0:single_lap_N]
        first_lap_vx = results.vx[0:single_lap_N]
        first_lap_vy = results.vy[0:single_lap_N]
        first_lap_r = results.r[0:single_lap_N]
        first_lap_st = results.st[0:single_lap_N]
        first_lap_T = results.T[0:single_lap_N]
        self.first_lap_global_refference  = Results(first_lap_delta_st, first_lap_delta_T, first_lap_n, first_lap_mu, first_lap_vx, first_lap_vy, first_lap_r, first_lap_st, first_lap_T, None, None, None, None)
                
        second_lap_delta_st = results.delta_st[single_lap_N:two_lap_N]
        second_lap_delta_T = results.delta_T[single_lap_N:two_lap_N]
        second_lap_n = results.n[single_lap_N:two_lap_N]
        second_lap_mu = results.mu[single_lap_N:two_lap_N]
        second_lap_vx = results.vx[single_lap_N:two_lap_N]
        second_lap_vy = results.vy[single_lap_N:two_lap_N]
        second_lap_r = results.r[single_lap_N:two_lap_N]
        second_lap_st = results.st[single_lap_N:two_lap_N]
        second_lap_T = results.T[single_lap_N:two_lap_N]
        self.first_lap_global_refference  = Results(second_lap_delta_st, second_lap_delta_T, second_lap_n, second_lap_mu, second_lap_vx, second_lap_vy, second_lap_r, second_lap_st, second_lap_T, None, None, None, None)


        print("Autonomous Navigation plan ready\n")
        self.have_plan = True
    
    def get_plan_from_csv(self, path):
        with open(path) as f:
            results = [tuple(line) for line in csv.reader(f)]
            for i in range(1, len(results)):
                pt = results[i]
                array_results = np.array([(float(pt[0]), (float(pt[1])), (float(pt[2])), (float(pt[3])), (float(pt[4])), (float(pt[5])), (float(pt[6])), (float(pt[7])), (float(pt[8])))])
            res = Results(array_results[1:, 0], array_results[1:, 1], array_results[1:, 2], array_results[1:, 3], array_results[1:, 4], array_results[1:, 5], array_results[1:, 6], array_results[1:, 7], array_results[1:, 8], None, None, None, None)      
        return res
 
    def get_action_from_plan(self):
        if self.have_plan:
            pose_x = self.position[0]
            pose_y = self.position[1]
            pose_theta = self.theta

            s, n = self.transform_XY_to_NS(pose_x, pose_y)

            if (self.previous_s - s) >= ((self.global_track.N-3) * self.global_track.delta_s):
                self.laps += 1
            self.previous_s = s
            steering_angle = self.stanley_control(s, n, pose_theta, self.long_vel, self.lat_vel)
            speed = self.speed_control(s)
        
        else:
            speed = 0
            steering_angle = 0

        return speed, steering_angle

    def odom_callback(self, msg, na):
        position = msg.pose.pose.position
        self.position = np.array([(position.x - self.global_track.origin[0]), (position.y - self.global_track.origin[1])])
        
        self.long_vel = msg.twist.twist.linear.x
        self.lat_vel = msg.twist.twist.linear.y
        if np.abs(self.long_vel) < 0.15:
            self.r_z = 0
        else:
            self.r_z = msg.twist.twist.angular.z
        
        roll, pitch, yaw = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        roll_radians = roll * np.pi / 180
        pitch_radians = pitch * np.pi / 180
        yaw_radians = yaw * np.pi / 180
        self.theta = copy.copy(yaw_radians)
        
        header_time_secs = msg.header.stamp.secs
        header_time_nsecs = msg.header.stamp.nsecs

        if self.prev_pf_odom_msg is None:
            self.test_time = time()
            self.odom_callback_time_nsecs =  header_time_nsecs
            self.odom_callback_time_secs = header_time_secs
            odom = Odometry()
            odom.header.frame_id = "/map"
            odom.child_frame_id = "ego_racecar/baselink"
            odom.header.stamp.secs = header_time_secs
            odom.header.stamp.nsecs = header_time_nsecs
            odom.pose.pose.position.x = 0
            odom.pose.pose.position.y = 0
            odom.pose.pose.orientation.z = 0
            odom.pose.pose.orientation.x = 0
            odom.pose.pose.orientation.y = 0
            odom.pose.pose.orientation.w = 1
            self.propagated_odom_pub.publish(odom)  
        else:
            if (header_time_nsecs - self.odom_callback_time_nsecs) < 0: 
                delta_t = ((header_time_nsecs - self.odom_callback_time_nsecs) * (1e-9)) + 1 - 0.0029
            else:
                delta_t = (header_time_nsecs - self.odom_callback_time_nsecs) * (1e-9) - 0.0029

            # if delta_t<0:
            #     print(delta_t)
            long_distance = (self.long_vel * delta_t) 
            self.propagated_x =  self.propagated_x + (long_distance*np.cos(self.propagated_theta))
            self.propagated_y =  self.propagated_y + (long_distance*np.sin(self.propagated_theta))
            self.propagated_theta =  self.propagated_theta + (self.r_z * delta_t)
            qx, qy, qz, qw = self.get_quaternion_from_euler(roll_radians, pitch_radians, self.propagated_theta)

            # self.propagated_x = self.prev_pf_odom_msg.pose.pose.position.x + ((self.long_vel * delta_t) * np.cos(self.prev_pf_odom_msg.pose.pose.orientation.z))
            # self.propagated_y = self.prev_pf_odom_msg.pose.pose.position.y + ((self.long_vel * delta_t )* np.sin(self.prev_pf_odom_msg.pose.pose.orientation.z))
            # qx, qy, qz, qw = self.get_quaternion_from_euler(0, 0, (self.prev_pf_odom_msg.pose.pose.orientation.z+(self.r_z*delta_t)))
            # self.propagated_theta = qz
            
            
            self.odom_callback_time_nsecs = header_time_nsecs
            self.odom_callback_time_secs = header_time_secs
            self.test_time = time()

            odom = Odometry()
            odom.header.frame_id = "/map"
            odom.child_frame_id = "ego_racecar/baselink"
            odom.header.stamp.secs = header_time_secs
            odom.header.stamp.nsecs = header_time_nsecs
            odom.pose.pose.position.x = self.propagated_x
            odom.pose.pose.position.y = self.propagated_y
            odom.pose.pose.orientation.x = 0
            odom.pose.pose.orientation.y = 0
            odom.pose.pose.orientation.z = qz# self.propagated_theta
            odom.pose.pose.orientation.w =  self.prev_pf_odom_msg.pose.pose.orientation.w # msg.pose.pose.orientation.w
            odom.twist.twist.linear.x = msg.twist.twist.linear.x
            odom.twist.twist.angular.z = msg.twist.twist.angular.z
            self.propagated_odom_pub.publish(odom)


    def pf_odom_callback(self, msg, na):
        self.prev_pf_odom_msg = msg

    
    def scan_callback(self, scan_msg):
        speed, steering_angle = self.get_action_from_plan()
        drive = AckermannDriveStamped()
        drive.drive.speed = speed
        drive.drive.steering_angle = steering_angle
        self.drive = drive
        self.drive_pub.publish(drive)

        

    def linearly_interpolate(self, yA, yB, xA, xB, xC):
        gradient = (yB - yA)/(xB - xA)
        diff = xC - xA
        value = (gradient*diff) + yA
        return value

    def track_curvarture(self, s):
        if s <= (self.track.delta_s * self.track.N):
            for i in range(self.previous_index, (self.track.N-1)):
                if ( (self.track.delta_s * i ) <= s) and ( (self.track.delta_s * (i+1) ) >= s):
                    curvature = self.linearly_interpolate(self.track.curvature[i] , self.track.curvature[i+1], (self.track.delta_s*i), (self.track.delta_s*(i+1)), s)
                    return curvature
            if ((self.track.delta_s *  self.track.N) >= s) and ( (self.track.delta_s * (self.track.N-1) ) <= s):
                    curvature = self.linearly_interpolate(self.track.curvature[self.track.N-1] , self.track.curvature[0], (self.track.delta_s*(self.track.N-1)), (self.track.delta_s*(self.track.N)), s)
                    return curvature
        raise Exception("Unable to find track curvature at current position")

    def find_angle(self, A, B, C):
        vec_AB = A - B
        vec_AC = A - C 
        dot = (A[0] - C[0])*(A[0] - B[0]) + (A[1] - C[1])*(A[1] - B[1])
        magnitude_AB = np.linalg.norm(vec_AB)
        magnitude_AC = np.linalg.norm(vec_AC)
        if magnitude_AB*magnitude_AC == 0:
            angle = np.arccos(dot/(1e-12))
        else:
            angle = np.arccos(dot/(magnitude_AB*magnitude_AC))
        return angle
    
    def transform_XY_to_NS(self, x, y):
        dx = x - self.global_track.track_points[0, :]
        dy = y - self.global_track.track_points[1, :]
        distances = np.hypot(dx, dy)

        ind_s = np.argmin(distances)
        if ind_s >= 1 and ind_s < (self.global_track.N-1):
            if distances[ind_s+1] <= distances[ind_s-1]:
                point_1 = np.array([self.global_track.track_points[0, ind_s], self.global_track.track_points[1, ind_s]])
                point_2 = np.array([self.global_track.track_points[0, ind_s+1], self.global_track.track_points[1, ind_s+1]])
                ang = self.find_angle(point_1, point_2, np.array([x, y]))
                actual_s = (ind_s * self.global_track.delta_s) + (np.cos(ang) * distances[ind_s])
                actual_n = (np.sin(ang) * distances[ind_s])
            
            elif  distances[ind_s+1] > distances[ind_s-1]:
                point_1 = np.array([self.global_track.track_points[0, ind_s], self.global_track.track_points[1, ind_s]])
                point_2 = np.array([self.global_track.track_points[0, ind_s-1], self.global_track.track_points[1, ind_s-1]])
                ang = self.find_angle(point_2, point_1, np.array([x, y]))
                actual_s = ((ind_s-1) * self.global_track.delta_s) + (np.cos(ang) * distances[ind_s-1])
                actual_n = (np.sin(ang) * distances[ind_s-1])
            else:
                raise Exception("Error")

        elif ind_s == 0:
            if distances[1] < distances[self.global_track.N-1]: 
                # Case where the car is at the beginning of the track
                point_1 = np.array([self.global_track.track_points[0, ind_s], self.global_track.track_points[1, ind_s]])
                point_2 = np.array([self.global_track.track_points[0, ind_s+1], self.global_track.track_points[1, ind_s+1]])
                ang = self.find_angle(point_1, point_2, np.array([x, y]))
                actual_s = (ind_s * self.global_track.delta_s) + (np.cos(ang) * distances[ind_s])
                actual_n = (np.sin(ang) * distances[ind_s])
            elif distances[1] > distances[self.global_track.N-1]: 
                # Case where the car is at the End of the track
                point_1 = np.array([self.global_track.track_points[0, self.global_track.N-1], self.global_track.track_points[1, self.global_track.N-1]])
                point_2 = np.array([self.global_track.track_points[0, ind_s], self.global_track.track_points[1, ind_s]])
                ang = self.find_angle(point_1, point_2, np.array([x, y]))
                actual_s = ((self.global_track.N-1) * self.global_track.delta_s) + (np.cos(ang) * distances[self.global_track.N-1])
                actual_n = (np.sin(ang) * distances[self.global_track.N-1])
        
        elif ind_s == self.global_track.N-1:
            if distances[0] < distances[ind_s]: 
                # Case where the car is past the last track discretization but not yet started a new lap
                point_1 = np.array([self.global_track.track_points[0, ind_s], self.global_track.track_points[1, ind_s]])
                point_2 = np.array([self.global_track.track_points[0, 0], self.global_track.track_points[1, 0]])
                ang = self.find_angle(point_1, point_2, np.array([x, y]))
                actual_s = (ind_s * self.global_track.delta_s) + (np.cos(ang) * distances[ind_s])
                actual_n = (np.sin(ang) * distances[ind_s])
            elif distances[0] > distances[self.global_track.N-1]: 
                # Case where the car is at the end of the track but before the last discretization point
                point_1 = np.array([self.global_track.track_points[0, ind_s-1], self.global_track.track_points[1, ind_s-1]])
                point_2 = np.array([self.global_track.track_points[0, ind_s], self.global_track.track_points[1, ind_s]])
                ang = self.find_angle(point_1, point_2, np.array([x, y]))
                actual_s = ((ind_s-1) * self.global_track.delta_s) + (np.cos(ang) * distances[ind_s-1])
                actual_n = (np.sin(ang) * distances[ind_s-1])
        
        xy_angle = np.arctan2((y-self.global_track.track_points[1, ind_s]),(x-self.global_track.track_points[0, ind_s]))   #angle between (x,y) and (s,0)
        yaw_angle = self.global_track.yaw_angle[ind_s]                               #angle at s
        angle = func.sub_angles_complex(xy_angle, yaw_angle)    
        if angle >=0:   #Vehicle is above s line
            direct=1    #Positive n direction
        else:           #Vehicle is below s line
            direct=-1   #Negative n direction

        actual_n = actual_n*direct   #Include sign
        
        return (actual_s, actual_n)   
    
    def transform_NS_to_XY(self, s, n):
        for i in range((self.global_track.N-1)):
            if ( (self.global_track.delta_s * i ) <= s) and ( (self.global_track.delta_s * (i+1) ) >= s):
                yaw_angle = self.linearly_interpolate(self.global_track.yaw_angle[i], self.global_track.yaw_angle[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                track_x = self.linearly_interpolate(self.global_track.track_points[0, i], self.global_track.track_points[0, i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                track_y = self.linearly_interpolate(self.global_track.track_points[1, i], self.global_track.track_points[1, i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                break
        actual_X = -n*np.sin(yaw_angle) + track_x
        actual_Y = n*np.cos(yaw_angle) + track_y

        return (actual_X, actual_Y)
    
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return qx, qy, qz, qw

    def stanley_control(self, s, n, mu, vx, vy):
        
        ref_orthogonal_distance = 0
        ref_local_heading = 0
        track_heading = 0


        if self.laps == 0:
            if ( (self.global_track.delta_s * (self.global_track.N-1)) <= s) and ( (self.global_track.delta_s * self.global_track.N ) >= s):
                ref_orthogonal_distance = self.linearly_interpolate(self.first_lap_global_refference .n[self.global_track.N-1], self.first_lap_global_refference .n[0], (self.global_track.delta_s*(self.global_track.N-1)), (self.global_track.delta_s*(self.global_track.N)), s)
                ref_local_heading = self.linearly_interpolate(self.first_lap_global_refference .mu[self.global_track.N-1], self.first_lap_global_refference .mu[0], (self.global_track.delta_s*(self.global_track.N-1)), (self.global_track.delta_s*(self.global_track.N)), s) 
                track_heading = self.linearly_interpolate(self.global_track.yaw_angle[self.global_track.N-1], self.global_track.yaw_angle[0], (self.global_track.delta_s*(self.global_track.N-1)), (self.global_track.delta_s*(self.global_track.N)), s)          
                self.previous_index = self.global_track.N-1
            else:
                for i in range((self.global_track.N-1)):
                    if ( (self.global_track.delta_s * i ) <= s) and ( (self.global_track.delta_s * (i+1) ) >= s):
                        ref_orthogonal_distance = self.linearly_interpolate(self.first_lap_global_refference .n[i], self.first_lap_global_refference .n[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                        ref_local_heading = self.linearly_interpolate(self.first_lap_global_refference .mu[i], self.first_lap_global_refference .mu[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s) 
                        track_heading = self.linearly_interpolate(self.global_track.yaw_angle[i], self.global_track.yaw_angle[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s) 
                        self.previous_index = i
                        break
            k = self.Stanley_Gain
        else: 
            if ( (self.global_track.delta_s * (self.global_track.N-1)) <= s) and ( (self.global_track.delta_s * self.global_track.N ) >= s):
                ref_orthogonal_distance = self.linearly_interpolate(self.first_lap_global_refference .n[self.global_track.N-1], self.first_lap_global_refference .n[0], (self.global_track.delta_s*(self.global_track.N-1)), (self.global_track.delta_s*(self.global_track.N)), s)
                ref_local_heading = self.linearly_interpolate(self.first_lap_global_refference .mu[self.global_track.N-1], self.first_lap_global_refference .mu[0], (self.global_track.delta_s*(self.global_track.N-1)), (self.global_track.delta_s*(self.global_track.N)), s) 
                track_heading = self.linearly_interpolate(self.global_track.yaw_angle[self.global_track.N-1], self.global_track.yaw_angle[0], (self.global_track.delta_s*(self.global_track.N-1)), (self.global_track.delta_s*(self.global_track.N)), s)          
                self.previous_index = self.global_track.N-1
            else:
                for i in range((self.global_track.N-1)):
                    if ( (self.global_track.delta_s * i ) <= s) and ( (self.global_track.delta_s * (i+1) ) >= s):
                        ref_orthogonal_distance = self.linearly_interpolate(self.first_lap_global_refference .n[i], self.first_lap_global_refference .n[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                        ref_local_heading = self.linearly_interpolate(self.first_lap_global_refference .mu[i], self.first_lap_global_refference .mu[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s) 
                        track_heading = self.linearly_interpolate(self.global_track.yaw_angle[i], self.global_track.yaw_angle[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s) 
                        self.previous_index = i
                        break
            k = self.Stanley_Gain
            
        heading_refferenced_to_x = func.normalize_angle(mu + self.orientation_x)
        heading_refferenced_to_track = func.normalize_angle(heading_refferenced_to_x) - func.normalize_angle(track_heading)
        
        normalised_heading =  func.normalize_angle(heading_refferenced_to_track)
        heading_error = ref_local_heading - normalised_heading

        cross_track_error = ref_orthogonal_distance - n
        cross_track_error = ref_orthogonal_distance - n
        velocity = np.sqrt((vx**2) + (vy**2) )
        cross_track_correction = np.arctan((k*cross_track_error)/(velocity+0.00001))
        

        stanley_control = heading_error + cross_track_correction

        if stanley_control < (-np.pi/3):
            stanley_control = -np.pi/3
        if stanley_control > (np.pi/3):
            stanley_control = np.pi/3
        return stanley_control

    def speed_control(self, s):
        speed = -1
        if self.laps == 0:
            if ( (self.global_track.delta_s * (self.global_track.N-1)) <= s) and ( (self.global_track.delta_s * self.global_track.N ) >= s):
                speed = self.linearly_interpolate(self.first_lap_global_refference .vx[self.global_track.N-1], self.first_lap_global_refference .vx[0], (self.global_track.delta_s*(self.global_track.N-1)), (self.global_track.delta_s*(self.global_track.N)), s)
                return speed
            for i in range((self.global_track.N-1)):
                if ( (self.global_track.delta_s * i ) <= s) and ( (self.global_track.delta_s * (i+1) ) >= s):
                    speed = self.linearly_interpolate(self.first_lap_global_refference .vx[i], self.first_lap_global_refference .vx[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                    return speed
        else:
            if ( (self.global_track.delta_s * (self.global_track.N-1)) <= s) and ( (self.global_track.delta_s * self.global_track.N ) >= s):
                speed = self.linearly_interpolate(self.first_lap_global_refference .vx[self.global_track.N-1], self.first_lap_global_refference .vx[0], (self.global_track.delta_s*(self.global_track.N-1)), (self.global_track.delta_s*(self.global_track.N)), s)
                return speed
            for i in range((self.global_track.N-1)):
                if ( (self.global_track.delta_s * i ) <= s) and ( (self.global_track.delta_s * (i+1) ) >= s):
                    speed = self.linearly_interpolate(self.first_lap_global_refference .vx[i], self.first_lap_global_refference .vx[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                    return speed
        if speed == -1:
            raise Exception("Speed Control Not Succesful")
    
    def get_horizon_track(self, scan):
        # is there an obstacle?
        # Yes: 
        #       From the current mu, n, and s, localise the obstacle from the scan in track cordinates
        #       Get the width of the ostacle
        #       Get the curvature and widths of the track from the global track from the car to the horizon distance
        #       Create a temporary track 
        # No: 
        #       Get the curvature and widths of the track from the global track from the car to the horizon distance
        #       Create a temporary track 
        
        pass
    def short_term_plan(self, temp_track, obstacle):
        pass

if __name__ == '__main__':
    rospy.init_node('sim_agent')
    dummy_agent = My_Agent('columbia_small')
    rospy.spin()

