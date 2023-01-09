import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import image
import math
import cmath
import yaml
from argparse import Namespace
import bisect
import sys
import cubic_spline_planner
import yaml
from PIL import Image, ImageOps, ImageDraw
import random
from datetime import datetime
import time
from numba import njit
from numba import int32, int64, float32, float64,bool_    
from numba import jitclass
#from numba.experimental import jitclass
import pickle

def load_config(path, fname):
    full_path = path + '/config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf

def add_angles(a1, a2):
    angle = (a1+a2)%(2*np.pi)

    return angle

def sub_angles(a1, a2):
    angle = (a1-a2)%(2*np.pi)

    return angle

def add_angles_complex(a1, a2):
    real = math.cos(a1) * math.cos(a2) - math.sin(a1) * math.sin(a2)
    im = math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

def add_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return ret


def get_distance(x1=[0, 0], x2=[0, 0]):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)
     
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret



def get_gradient(x1=[0, 0], x2=[0, 0]):
    t = (x1[1] - x2[1])
    b = (x1[0] - x2[0])
    if b != 0:
        return t / b
    return 1000000 # near infinite gradient. 

def transform_coords(x=[0, 0], theta=np.pi):
    # i want this function to transform coords from one coord system to another
    new_x = x[0] * np.cos(theta) - x[1] * np.sin(theta)
    new_y = x[0] * np.sin(theta) + x[1] * np.cos(theta)

    return np.array([new_x, new_y])

def normalise_coords(x=[0, 0]):
    r = x[0]/x[1]
    y = np.sqrt(1/(1+r**2)) * abs(x[1]) / x[1] # carries the sign
    x = y * r
    return [x, y]

def get_bearing(x1=[0, 0], x2=[0, 0]):
    grad = get_gradient(x1, x2)
    dx = x2[0] - x1[0]
    th_start_end = np.arctan(grad)
    if dx == 0:
        if x2[1] - x1[1] > 0:
            th_start_end = 0
        else:
            th_start_end = np.pi
    elif th_start_end > 0:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = -np.pi/2 - th_start_end
    else:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = - np.pi/2 - th_start_end

    return th_start_end


def normalize_angle_positive(angle):
    """
    Wrap the angle between 0 and 2 * pi.
    Args:
        angle (float): angle to wrap.
    Returns:
         The wrapped angle.
    """
    pi_2 = 2. * np.pi
    return math.fmod(math.fmod(angle, pi_2) + pi_2, pi_2) 

def normalize_angle(angle):
    """
    Wrap the angle between -pi and pi.
    Args:
        angle (float): angle to wrap.
    Returns:
         The wrapped angle.
    """
    a = normalize_angle_positive(angle)
    if a > np.pi:
        a -= 2. * np.pi
    return a 


#@njit(cache=True)
def distance_between_points(x1, x2, y1, y2):
    distance = math.hypot(x2-x1, y2-y1)
    return distance


def generate_circle_image():
    from matplotlib import image
    image = Image.new('RGBA', (600, 600))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 600, 600), fill = 'black', outline ='black')
    draw.ellipse((50, 50, 550, 550), fill = 'white', outline ='white')
    draw.ellipse((150, 150, 450, 450), fill = 'black', outline ='black')
    draw.point((100, 100), 'red')
    image_path = sys.path[0] + '\\maps\\circle' + '.png'
    image.save(image_path, 'png')


def generate_circle_goals():
    from matplotlib import image
    #image_path = sys.path[0] + '\\maps\\circle' + '.png'
    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))

    R=10
    theta=np.linspace(0, 2*math.pi, 17)
    x = 15+R*np.cos(theta-math.pi/2)
    y = 15+R*np.sin(theta-math.pi/2)
    rx, ry, ryaw, rk, s = cubic_spline_planner.calc_spline_course(x, y)
    #plt.plot(rx, ry, "-r", label="spline")
    #plt.plot(x, y, 'x')
    #plt.show()
    return x, y, rx, ry, ryaw, rk, s


def generate_berlin_goals():
    from matplotlib import image
    #image_path = sys.path[0] + '/maps/berlin' + '.png'
    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))
    
    goals = [[16,3], [18,4], [18,7], [18,10], [18.5, 13], [19.5,16], [20.5,19], [19.5,22], [17.5,24.5], 
            [15.5,26], [13,26.5], [10,26], [7.5,25], [6,23], [7,21.5], [9.5,21.5], [11, 21.5], 
            [11,20], [10.5,18], [11,16], [12,14], [13,12], [13.5,10], [13.5,8], [14,6], [14.5,4.5], [16,3]]
    
    x = []
    y = []

    for xy in goals:
        x.append(xy[0])
        y.append(xy[1])
    
    rx, ry, ryaw, rk, s = cubic_spline_planner.calc_spline_course(x, y)

    #plt.plot(rx, ry, "-r", label="spline")
    #plt.plot(x, y, 'x')
    #plt.show()

    return x, y, rx, ry, ryaw, rk, s
    

def map_generator(map_name):
    map_config_path = sys.path[0] + '/maps/' + map_name + '.yaml'
    image_path = sys.path[0] + '/maps/' + map_name + '.png'
    with open(map_config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    map_conf = Namespace(**conf_dict)
    
    res=map_conf.resolution

    with Image.open(image_path) as im:
        gray_im = ImageOps.grayscale(im)
        map_array = np.asarray(gray_im)
        map_height = gray_im.height*res
        map_width = gray_im.width*res
        occupancy_grid = map_array<1
    
    return occupancy_grid, map_height, map_width, res 

   
def random_start(x, y, rx, ry, ryaw, rk, s, episode):
    
    offset=0.3
    random.seed(datetime.now())
    '''
    if episode < 20000:
        if random.uniform(0,1)<0.1:
            i = int(random.uniform(0, len(x)-2))
        else:
            i = int(random.uniform(10, 14))
    
    elif episode >= 20000 and episode <50000:
        if random.uniform(0,1)<0.5:
            i = int(random.uniform(0, len(x)-2))
        else:
            i = int(random.uniform(10, 14))
    else:
    '''
    i = int(random.uniform(0, len(x)-2))
    
    #i = int(random.uniform(0, len(x)-2))
    #i = int(random.uniform(10, 12))
    
    next_i = (i+1)%len(y)
    start_x = x[i] + (random.uniform(-offset, offset))
    start_y = y[i] + (random.uniform(-offset, offset))
    
    start_theta = math.atan2(y[next_i]-y[i], x[next_i]-x[i]) + (random.uniform(-math.pi/6, math.pi/6))
    next_goal = (i+1)%len(x)

    return start_x, start_y, start_theta, next_i


def find_closest_point(rx, ry, x, y):

    dx = [x - irx for irx in rx]
    dy = [y - iry for iry in ry]
    d = np.hypot(dx, dy)    
    ind = np.argmin(d)
    
    return ind



def find_angle_to_line(ryaw, theta):

    angle = np.abs(sub_angles_complex(ryaw, theta))

    return angle


@njit(cache=True)
def occupied_cell(x, y, occupancy_grid, res, map_height):
    
    cell = (np.array([map_height-y, x])/res).astype(np.int64)

    if occupancy_grid[cell[0], cell[1]] == True:
        return True
    else:
        return False


spec = [('lidar_res', float32),
        ('n_beams', int32),
        ('max_range', float32),
        ('fov', float32),
        ('occupancy_grid', bool_[:,:]),
        ('map_res', float32),
        ('map_height', float32),
        ('beam_angles', float64[:])]


@jitclass(spec)
class lidar_scan():
    def __init__(self, lidar_res, n_beams, max_range, fov, occupancy_grid, map_res, map_height):
        
        self.lidar_res = lidar_res
        self.n_beams  = n_beams
        self.max_range = max_range
        self.fov = fov
        
        #self.beam_angles = (self.fov/(self.n_beams-1))*np.arange(self.n_beams)
        
        self.beam_angles = np.zeros(self.n_beams, dtype=np.float64)
        for n in range(self.n_beams):
            self.beam_angles[n] = (self.fov/(self.n_beams-1))*n

        self.occupancy_grid = occupancy_grid
        self.map_res = map_res
        self.map_height = map_height

    def get_scan(self, x, y, theta):
        
        scan = np.zeros((self.n_beams))
        coords = np.zeros((self.n_beams, 2))
        
        for n in range(self.n_beams):
            i=1
            occupied=False

            while i<(self.max_range/self.lidar_res) and occupied==False:
                x_beam = x + np.cos(theta+self.beam_angles[n]-self.fov/2)*i*self.lidar_res
                y_beam = y + np.sin(theta+self.beam_angles[n]-self.fov/2)*i*self.lidar_res
                occupied = occupied_cell(x_beam, y_beam, self.occupancy_grid, self.map_res, self.map_height)
                i+=1
            
            coords[n,:] = [np.round(x_beam,3), np.round(y_beam,3)]
            #dist = np.linalg.norm([x_beam-x, y_beam-y])
            dist = math.sqrt((x_beam-x)**2 + (y_beam-y)**2)
            
            scan[n] = np.round(dist,3)

        return scan, coords

def generate_initial_condition(name, episodes):
   file_name = 'test_initial_condition/' + name
   
   initial_conditions = []
   
   goal_x, goal_y, rx, ry, ryaw, rk, d = generate_circle_goals()
   
   for eps in range(episodes):
      x, y, theta, current_goal = random_start(goal_x, goal_y, rx, ry, ryaw, rk, d, episode=0)
      v = random.random()*7
      delta = 0
      i = {'x':x, 'y':y, 'v':v, 'delta':delta, 'theta':theta, 'goal':current_goal}
      initial_conditions.append(i)

   #initial_conditions = [ [] for _ in range(episodes)]
   
   outfile=open(file_name, 'wb')
   pickle.dump(initial_conditions, outfile)
   outfile.close()

#generate_berlin_goals()
if __name__ == 'main':
    #def velocity_along_line(theta, velocity, ryaw, )

    #generate_berlin_goals()
    #x, y, rx, ry, ryaw, rk, s = generate_circle_goals()
    #start_x, start_y, start_theta, next_goal = random_start(x, y, rx, ry, ryaw, rk, s)

    #image_path = sys.path[0] + '/maps/' + 'circle' + '.png'       
    #occupancy_grid, map_height, map_width, res = map_generator(map_name='circle')
    #a = lidar_scan(res, 3, 10, np.pi, occupancy_grid, res, 30)
    #print(a.get_scan(15,5,0))


    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))
    #plt.plot(start_x, start_y, 'x')
    #print(start_theta)
    #plt.arrow(start_x, start_y, math.cos(start_theta), math.sin(start_theta))
    #plt.plot(x, y, 's')
    #plt.plot(x[next_goal], y[next_goal], 'o')
    #plt.show()
    pass
