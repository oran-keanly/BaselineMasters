from turtle import distance
import numpy as np
from mapping import map
import functions as func
from matplotlib import  pyplot as plt
from optimization import Kinematic_Bicycle_Model_States
#from scripts.my_agent import Agent
from track import Track
from parameters import Kinematic_Bicycle_Model_Parameters
from optimization import Results
from optimization import Optimizer
from scipy.integrate import odeint

class auto_car:
    
    def __init__(self, global_track, initialConditions=None, model_parameters=None, sample_time=0.1, path=None):
        self.global_track = global_track
        if initialConditions is None:
            self.initialConditions = Kinematic_Bicycle_Model_States(0, 0, 0, 0, 5, 0, 0, 0, 0) # my Model; 
        else:
            self.initialConditions = initialConditions
        
        if model_parameters is None:
            self.model_parameters = Kinematic_Bicycle_Model_Parameters(Slip_AngleReg=5, Steer_Reg=5, Max_Tyre_Force=10.0)
        else:
            self.model_parameters = model_parameters
        
        self.opt = Optimizer(self.global_track, self.model_parameters, print_level=5)
        
        if path is not None:
            self.Global_Reference = self.opt.optimize(self.initialConditions, export_results=True, save_path=path)
        else:
            self.Global_Reference = self.opt.optimize(self.initialConditions)
        
        self.get_gloabl_xy_reference()
        self.plot_global_reference()
        self.sample_time = sample_time
        self.savePath = path
        self.Stanley_Gain = 5

    def linearly_interpolate(self, yA, yB, xA, xB, xC):
        gradient = (yB - yA)/(xB - xA)
        diff = xC - xA
        value = (gradient*diff) + yA
        return value

    def track_curvarture(self, s):
        if s <= (self.track.delta_s * self.track.N):
            for i in range((self.track.N)-1):
                if ( (self.track.delta_s * i ) <= s) and ( (self.track.delta_s * (i+1) ) >= s):
                    curvature = self.linearly_interpolate(self.track.curvature[i] , self.track.curvature[i+1], (self.track.delta_s*i), (self.track.delta_s*(i+1)), s)
                    return curvature
            if ( (self.track.delta_s * self.track.N ) >= s) and ( (self.track.delta_s * (self.track.N-1) ) <= s):
                curvature = self.linearly_interpolate(self.track.curvature[self.track.N-1] , self.track.curvature[0], (self.track.delta_s*(self.track.N -1) ), (self.track.delta_s*self.track.N), s)
                return curvature  
        raise Exception("Unable to find track curvature at current position")

    def find_angle(self, A, B, C):
        # RETURNS THE ANGLE BÂC
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
            if distances[ind_s+1] < distances[ind_s-1]:
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
            point_1 = np.array([self.global_track.track_points[0, ind_s], self.global_track.track_points[1, ind_s]])
            point_2 = np.array([self.global_track.track_points[0, ind_s+1], self.global_track.track_points[1, ind_s+1]])
            ang = self.find_angle(point_1, point_2, np.array([x, y]))
            actual_s = (ind_s * self.global_track.delta_s) + (np.cos(ang) * distances[ind_s])
            actual_n = (np.sin(ang) * distances[ind_s])
        elif ind_s == self.global_track.N-1:
            point_1 = np.array([self.global_track.track_points[0, ind_s], self.global_track.track_points[1, ind_s]])
            point_2 = np.array([self.global_track.track_points[0, ind_s-1], self.global_track.track_points[1, ind_s-1]])
            ang = self.find_angle(point_2, point_1, np.array([x, y]))
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
    
    def get_gloabl_xy_reference(self):
        optimal_xy_path = np.zeros((2, self.global_track.N))
        for i in range(self.global_track.N):
            x, y = self.transform_NS_to_XY(i*self.global_track.delta_s, self.Global_Reference.n[i])
            optimal_xy_path[0, i] = x
            optimal_xy_path[1, i] = y
        self.global_XY_reference =  optimal_xy_path
        return optimal_xy_path
    
    def plot_global_reference(self):
        plt.imshow(self.global_track.map_image, extent=(0,(self.global_track.map_width),0,(self.global_track.map_height)))
        plt.plot(self.global_XY_reference[0,:] , self.global_XY_reference[1,:] )
        plt.show()

    def car_control(self, state, sample_time):
        # state = numpy array                  input = numpy array                                            
        # state[0] = Centerline Distance       input[0] = Steering Angle
        # state[1] = Orthogonal Distance       input[1] = Driver Command
        # state[2] = Local Heading
        # state[3] = Longitudinal Velocity
        # state[4] = Lateral Velocity
        # state[5] = Yaw Rate
        if(state[0] <=0):
            raise Exception("Car has gone backwards")
        for i in range((self.global_track.N)-1):
            if ( (self.global_track.delta_s * i ) <= state[0]) and ( (self.global_track.delta_s * (i+1) ) >= state[0]):
                T = self.linearly_interpolate(self.Global_Reference.T[i], self.Global_Reference.T[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), state[0])
                steer = self.linearly_interpolate(self.Global_Reference.st[i], self.Global_Reference.st[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), state[0]) 
                # T_dot = self.linearly_interpolate(self.Global_Reference.delta_T[i], self.Global_Reference.delta_T[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), state[0])
                # steer_dot = self.linearly_interpolate(self.Global_Reference.delta_st[i], self.Global_Reference.delta_st[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), state[0]) 
                return np.array([self.stanley_control(state), T])
        return np.array([0, 0])

    def stanley_control(self, state):
        # state = numpy array                  input = numpy array                                            
        # state[0] = Centerline Distance       input[0] = Steering Angle
        # state[1] = Orthogonal Distance       input[1] = Driver Command
        # state[2] = Local Heading
        # state[3] = Longitudinal Velocity
        # state[4] = Lateral Velocity
        # state[5] = Yaw Rate
        if(state[0] <=0):
            raise Exception("Car has gone backwards")
        for i in range((self.global_track.N-1)):
            if ( (self.global_track.delta_s * i ) <= state[0]) and ( (self.global_track.delta_s * (i+1) ) >= state[0]):
                ref_orthogonal_distance = self.linearly_interpolate(self.Global_Reference.n[i], self.Global_Reference.n[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), state[0])
                ref_local_heading = self.linearly_interpolate(self.Global_Reference.mu[i], self.Global_Reference.mu[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), state[0]) 
                break
        k = self.Stanley_Gain
        normalised_heading =  func.normalize_angle(state[2])
        heading_error = ref_local_heading - normalised_heading
        cross_track_error = ref_orthogonal_distance - state[1]
        velocity = np.sqrt((state[3]**2) + (state[4]**2) )
        cross_track_correction = np.arctan((k*cross_track_error)/velocity)
        stanley_control = heading_error + cross_track_correction
        if stanley_control < (-np.pi/3):
            stanley_control = -np.pi/3
        if stanley_control > (np.pi/3):
            stanley_control = np.pi/3
        return stanley_control
   
    def update_local_plan(self, state, track):
        pass

    def approximate_lidar_scan(self):
        pass

    def state_approximation(self):
        pass

if __name__ == "__main__":
    
    # m = map('test_ESL_map')
    # m = map('berlin') 
    # track = m.generate_track(0.1)
    # plt.plot(track.curvature)
    # plt.show()
    # track.curvature = track.curvature * 1
    # plt.close()
    # plt.plot(track.curvature)
    # plt.show()
    # plt.close()
    
    # m = map('columbia_small') #Works
    # track = m.generate_track(0.1)
    # model_parameters = Kinematic_Bicycle_Model_Parameters(Slip_AngleReg=5, Steer_Reg=5, Max_Tyre_Force=3.0)
    # init_conditions = Kinematic_Bicycle_Model_States(0, 0, 0, 0, 5, 0, 0, 0, 0)
    # agent = auto_car(track, initialConditions=init_conditions, model_parameters=model_parameters, path='/home/oran/Documents')
    # #sim_env.main(agent)
    # m = map('columbia_small')
    # global_track = m.generate_track(0.1)
    # init_conditions = Kinematic_Bicycle_Model_States(0, 0, 0, 0, 0.5, 0, 0, 0, 0)
    # model_parameters = Kinematic_Bicycle_Model_Parameters(Slip_AngleReg=5, Steer_Reg=5, Max_Tyre_Force=5.0)
    # agent = auto_car(global_track=global_track, initialConditions=init_conditions, model_parameters=model_parameters, sample_time=0.1, path="./test/results")

    m = map('columbia_small')
    global_track = m.generate_track(0.1)
    single_track = global_track
    single_lap_N = len(single_track.curvature)
    track_twice = np.zeros(single_lap_N*2)
    two_lap_N = single_lap_N*2


    for i in range(single_lap_N):
        track_twice[i] = single_track.curvature[i]
    for i in range(single_lap_N, two_lap_N):
        track_twice[i] = single_track.curvature[i-single_lap_N]
    two_lap_track = Track(two_lap_N, 0.1, track_twice, 1.4)
    
    #global_track = Track(two_lap_N, 0.1, 0, 1.4)

    init_conditions = Kinematic_Bicycle_Model_States(0, 0, 0, 0, 1.0, 0, 0, 0, 0)
    model_parameters =  Kinematic_Bicycle_Model_Parameters(Slip_AngleReg=5, Steer_Reg=5, Max_Tyre_Force=15.0, Motor_gain=8, Friction_Ellipse=2, Drag_Resist=5)
    agent = auto_car(global_track=two_lap_track, initialConditions=init_conditions, model_parameters=model_parameters, sample_time=0.1, path="./Reference for columbia_small")