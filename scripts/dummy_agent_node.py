#!/usr/bin/env python
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import csv
import copy
from agent_utils import get_actuation, nearest_point_on_trajectory_py2, first_point_on_trajectory_intersecting_circle, quaternion_to_euler_angle
#from agent_utils import Trajectory

from mapping import map
import functions as func
from matplotlib import  pyplot as plt
from optimization import Kinematic_Bicycle_Model_States
from track import Track
from parameters import Kinematic_Bicycle_Model_Parameters
from optimization import Results
from optimization import Optimizer
from scipy.integrate import odeint

class Agent(object):
    def __init__(self, csv_path):
        # TODO: load waypoints from csv
        #self.trajectory = Trajectory('columbia_small')
        #self.waypoints = self.trajectory.waypoints
        #self.safe_speed = 0.5
        pass


class PurePursuitAgent(Agent):
    def __init__(self, csv_path, wheelbase):
        super(PurePursuitAgent, self).__init__(csv_path)
        self.lookahead_distance = 1.0
        self.wheelbase = wheelbase
        self.max_reacquire = 10.
        
        self.position = np.array([0, 0])
        self.theta = 0

        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback, 10)
        self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)

        with open(csv_path) as f:
            wpts = [tuple(line) for line in csv.reader(f)]
            self.waypoints = np.array([(float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3]), float(pt[4]), float(pt[5])) for pt in wpts])

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = waypoints[:, 0:2]
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty(waypoints[i2, :].shape)
            # x, y
            current_waypoint[0:2] = waypoints[i2, 0:2]
            # theta
            current_waypoint[3] = waypoints[i2, 3]
            # speed
            current_waypoint[2] = waypoints[i2, 2]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return waypoints[i, :]
        else:
            return None

    def plan(self, obs):
        pose_x = self.position[0]
        pose_y = self.position[1]
        pose_theta = self.theta
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, self.lookahead_distance, position, pose_theta)
        if lookahead_point is None:
            return self.safe_speed, 0.0
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, self.lookahead_distance, self.wheelbase)
        return speed, steering_angle

    def odom_callback(self, msg, na):
        position = msg.pose.pose.position
        self.position = np.array([position.x, position.y])
        self.velocity_x = msg.twist.twist.linear.x
        self.velocity_y = msg.twist.twist.linear.y

        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        theta = z * np.pi / 180
        self.theta = copy.copy(theta)


    def scan_callback(self, scan_msg):
        # print('got scan, now plan')
        speed, steering_angle = self.plan(None)
        drive = AckermannDriveStamped()
        drive.drive.speed = 2#speed
        drive.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive)

if __name__ == '__main__':
    rospy.init_node('dummy_agent')
    dummy_agent = PurePursuitAgent('/home/oran/catkin_ws/src/F1tenth/ros1/f1tenth_gym_ros/maps/columbia_small_std.csv', 1)
    rospy.spin()

