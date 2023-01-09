import yaml 
from argparse import Namespace
import math 
import numpy as np
import csv
from numba import njit


import os, shutil

EPSILON = 0.00000000001



@njit(fastmath=True, cache=True)
def add_locations(x1, x2, dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return ret

@njit(fastmath=True, cache=True)
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret

@njit(fastmath=False, cache=True)
def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# finds the nearest point on the given line segment connecting start and end
    # all arguments should be numpy arrays
def nearest_point_on_line_segment(point, start, end):
    '''
    Return the nearest point along the line segment connecting start and end.

    >>> nearest_point_on_line_segment(np.array([0.0,0.0]), np.array([1.0,1.0]), np.array([1.0,-1.0]))
    (array([ 1.,  0.]), 0.5)

    >>> nearest_point_on_line_segment(np.array([0.0,0.0]), np.array([1.0,1.0]), np.array([1.0,2.0]))
    (array([ 1.,  1.]), 0.0)

    >>> nearest_point_on_line_segment(np.array([0.0,0.0]), np.array([1.0,-2.0]), np.array([1.0,-1.0]))
    (array([ 1., -1.]), 1.0)
    '''
    diff = start - end
    l2 = np.dot(diff, diff)
    if l2 == 0.0:
        return start, 0.0
    t = np.clip(np.dot(point - start, end - start) / l2, 0.0, 1.0)
    projection = start + t * (end - start)
    return projection, t


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory_py2(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

    # print min_dist_segment, dists[min_dist_segment], projections[min_dist_segment]

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    # waypoint_car = np.dot(get_rotation_matrix(-pose_theta), (lookahead_point[0:2]-position))
    # waypoint_y = waypoint_car[1]
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle


# coords: Nx2 in polar (r,theta)
# in place modifies to Nx2 (x,y)
# def polar_to_euclid(coords):
#     xs = ranges * np.cos(angles)
#     ys = ranges * np.sin(angles)
#     return (xs, ys)

def angular_deflection_magnitude(points):
    # https://mail.python.org/pipermail/tutor/2007-July/055178.html
    # returns a numpy array of angular deflections between consequtive
    # line segments beginning and ending at the points provided
    # contains two less angles than points, since the angular deflection for the first and last components is ill defined

    lines = np.zeros((points.shape[0]-1, 3))
    thetas = np.zeros(points.shape[0]-2)
    for i in range(1,points.shape[0]):
        p0 = points[i-1,:]
        p1 = points[i,:]

        A = p0[1] - p1[1]
        B = p1[0] - p0[0]
        C = p0[0]*p1[1] - p1[0]*p0[1]
        lines[i-1] = (A,B,C)

    for i in range(1, lines.shape[0]):
        A1 = lines[i-1,0]
        B1 = lines[i-1,1]
        A2 = lines[i,0]
        B2 = lines[i,1]
        bottom = (A1**2+B1**2)*(A2**2+B2**2)
        if bottom > 0:
            inner = (A1*A2 + B1*B2) / np.sqrt(bottom)
            # thetas[i-1] = np.arccos(inner)
            if np.abs(np.abs(inner) - 1.0) < EPSILON:
                thetas[i-1] = 0.0
            else:
                thetas[i-1] = np.arccos(inner)
    return thetas

def piecewise_linear_local_waypoints_polar(points):
    thetas = angular_deflection_magnitude(points)
    # # compute the polar coordinate space local coordinate frame waypoints (r,theta)
    local_points_polar = np.zeros((points.shape[0]-1, 2))
    for i in range(1, points.shape[0]-1):
        # radius
        local_points_polar[i-1,0] = np.linalg.norm(points[i,:] - points[i-1,:])
        # angle
        local_points_polar[i,1] = thetas[i-1]
    local_points_polar[-1,0] = np.linalg.norm(points[-1,:] - points[-2,:])
    return local_points_polar

    # local_points_cartesian = np.zeros_like(local_points_polar)
    # local_points_cartesian[:,0] = local_points_polar[:,0] * np.cos(local_points_polar[:,1])
    # local_points_cartesian[:,1] = local_points_polar[:,0] * np.sin(local_points_polar[:,1])
    # print local_points_cartesian
    # print local_points_polar

###################################################################
# BEN"S UTIL FUNCTIONS

def load_conf(fname):
    # mac_path = "/Users/benjamin/Documents/GitHub/safety_system_ros/config/"

    full_path =  "/home/benjy/sim_ws/src/safety_system_ros/config/" + fname + '.yaml'
    # full_path =  "/home/nvidia/f1tenth_ws/src/safety_system_ros/config/" + fname + '.yaml'
    # full_path =  mac_path + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)
    # np.random.seed(conf.random_seed)
    return conf

def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)


def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z


def orientation_to_angle(orientation):
    x, y, z = quaternion_to_euler_angle(orientation.w, orientation.x, orientation.y, orientation.z)
    theta = z * np.pi / 180

    return theta


@njit(cache=True)
def limit_phi(phi):
    while phi > np.pi:
        phi = phi - 2*np.pi
    while phi < -np.pi:
        phi = phi + 2*np.pi
    return phi


@njit(cache=True)
def calculate_speed(delta):
    b = 0.523
    g = 9.81
    l_d = 0.329
    f_s = 0.8
    max_v = 4

    if abs(delta) < 0.06:
        return max_v

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    return V
####################################################################################################

class AckermannModel(object):
    """ A wrapper class for useful Ackermann steering geometry related functions
    """
    def __init__(self, wheelbase):
        self.L = wheelbase

    def path_radius(self, steering_angle):
        ''' The radius of the path driven if a constant steering angle is applied
        '''
        return self.L / np.tan(steering_angle)

    def yaw_rate(self, steering_angle, speed):
        ''' Rate of change of heading with a given steering angle and speed
        '''
        if steering_angle == 0.0:
            return 0.0
        return speed / self.path_radius(steering_angle)

    def dx(self, speed, dt, steering_angle):
        ''' Distance traveled in the local x direction given speed and steering_angle
        '''
        if steering_angle == 0.0:
            return speed * dt
        R = self.path_radius(steering_angle)
        d = dt*speed
        dx = R*np.sin(d/R)
        return dx

    def dy(self, speed, dt, steering_angle):
        ''' Distance traveled in the local y direction given speed and steering_angle
        '''
        if steering_angle == 0.0:
            return 0.0
        R = self.path_radius(steering_angle)
        d = dt*speed
        dy = R*(1.0 - np.cos(d/R))
        return dy

    def steering_angle(self, point):
        ''' Returns the steering angle required to pass through the given point
            (in local euclidean coordinates) assuming constant steering angle is applied
        '''
        theta = np.arctan2(point[1], point[0])
        return np.arctan(2.0*self.L*np.sin(theta)/np.linalg.norm(point))

    def steering_angle_polar(self, polar_point):
        ''' Returns the steering angle required to pass through the given point
            (in local polar coordinates) assuming constant steering angle is applied
        '''
        theta = polar_point[1]
        radius = polar_point[0]
        return np.arctan(2.0*self.L*np.sin(theta)/radius)

def max_angle(min_turning_radius, radius):
    tr2 = 2.0*min_turning_radius
    if radius < tr2:
        r2 = radius*radius
        y = r2 / (2.0*min_turning_radius)
        x = np.sqrt(r2 - y*y)
        max_angle = np.arctan(y/x)
    else:
        max_angle = np.pi / 2.0
    return max_angle


###############################################################################
# Trajectory Class from Ben's pp_utils.py

# class Trajectory:
#     def __init__(self, map_name):
#         self.map_name = map_name
#         self.waypoints = None
#         self.vs = None

#         self.max_reacquire = 20

#         self.diffs = None 
#         self.l2s = None 
#         self.ss = None 
#         self.o_points = None
#         self.traj_name = "Not set yet"
#         # self.load_csv_track()
#         # self.directory = f'/home/nvidia/f1tenth_ws/src/safety_system_ros/'
#         self.directory = f'/home/oran/catkin_ws/src/F1tenth/ros1/f1tenth_gym_ros/maps/'
#         self.load_centerline_csv_track()
#         self.n_wpts = len(self.waypoints)

#     def load_csv_track(self):
#         track = []
        
#         # filename = f'/home/benjy/sim_ws/src/safety_system_ros/map_data/{self.map_name}_centerline.csv'
#         # filename = os.path.dirname(os.path.abspath(__file__)) +  '/map_data/' + self.map_name + "_centerline.csv"
#         filename = self.directory + self.map_name + "_opti.csv"
#         with open(filename, 'r') as csvfile:
#             csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
#             for lines in csvFile:  
#                 track.append(lines)

#         track = np.array(track)
#         print(f"Track Loaded: {filename}")

#         # these get expanded
#         self.waypoints = track[:, 1:3]
#         # self.waypoints = track[:, 0:2]
#         self.vs = track[:, 5]
#         # self.vs = np.ones_like(track[:, 0])

#         # these don't get expanded
#         self.N = len(track)
#         self.o_pts = np.copy(self.waypoints)
#         self.ss = track[:, 0]
#         self.diffs = self.o_pts[1:,:] - self.o_pts[:-1,:]
#         self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 
#         self.show_trajectory()

        
#     def load_centerline_csv_track(self):
#         track = []
        
#         # filename = f'/home/benjy/sim_ws/src/safety_system_ros/map_data/{self.map_name}_centerline.csv'
#         # filename = os.path.dirname(os.path.abspath(__file__)) +  '/map_data/' + self.map_name + "_centerline.csv"
#         filename = self.directory + self.map_name + "_std.csv"
#         with open(filename, 'r') as csvfile:
#             csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
#             for lines in csvFile:  
#                 track.append(lines)

#         track = np.array(track)
#         print(f"Track Loaded: {filename}")

#         # these get expanded
#         self.waypoints = track[:, 0:2]
#         self.waypoints = self.waypoints[::-1, :]
#         self.vs = np.ones_like(track[:, 0]) * 1 

#         # these don't get expanded
#         self.N = len(track)
#         self.o_pts = np.copy(self.waypoints)
#         self.ss = track[:, 0]
#         self.diffs = self.o_pts[1:,:] - self.o_pts[:-1,:]
#         self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 
#         # self.show_trajectory()

#         self._expand_wpts()
#         self.traj_name = filename

#     def _expand_wpts(self):
#         n = 5 # number of pts per orig pt 
#         dz = 1 / n
#         o_line = self.waypoints
#         o_vs = self.vs
#         new_line = []
#         new_vs = []
#         for i in range(len(o_line)-1):
#             dd = sub_locations(o_line[i+1], o_line[i])
#             for j in range(n):
#                 pt = add_locations(o_line[i], dd, dz*j)
#                 new_line.append(pt)

#                 dv = o_vs[i+1] - o_vs[i]
#                 new_vs.append(o_vs[i] + dv * j * dz)

#         self.waypoints = np.array(new_line)
#         self.vs = np.array(new_vs)

#         # self.show_pts()

#     def get_current_waypoint(self, position, lookahead_distance):
#         #TODO: for compuational efficiency, pass the l2s and the diffs to the functions so that they don't have to be recalculated
#         wpts = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T
#         nearest_point, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, wpts)
#         if nearest_dist < lookahead_distance:
#             try:
#                 lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
#             except:
#                 i2 = 2
#             if i2 == None:
#                 return None
#             current_waypoint = np.empty((3, ))
#             # x, y
#             current_waypoint[0:2] = wpts[i2, :]
#             # speed
#             current_waypoint[2] = self.vs[i]
#             return current_waypoint
#         elif nearest_dist < self.max_reacquire:
#             return np.append(wpts[i, :], self.vs[i])
#         else:
#             raise Exception("Waypoint not found")

#     def show_pts(self):
        # from matplotlib import pyplot as plt
        # plt.figure(3)
        # plt.plot(self.waypoints[:,0], self.waypoints[:,1])
        # plt.gca().set_aspect('equal', adjustable='box')
        # # plt.show()
        # plt.pause(0.00001)