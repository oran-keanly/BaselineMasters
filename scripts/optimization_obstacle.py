# import libraries
# Pyomo stuff
from re import S
from statistics import NormalDist
from turtle import width
from pyomo.environ import*
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

from pyomo.environ import *
from pyomo.dae import *

# other
import sympy as sym
import numpy as np
import pandas as pd
import os
#from IPython.display import display #for pretty printing

import numpy as np
import matplotlib.pyplot as plt
import copy

from track import Obstacle, Track
from parameters import Kinematic_Bicycle_Model_Parameters

from mapping import map

class Kinematic_Bicycle_Model_States:
    def __init__(self, delta_st, delta_T, n, mu, vx, vy, r, st, T):
        self.delta_st = delta_st
        self.delta_T = delta_T
        self.n = n
        self.mu = mu
        self.vx = vx
        self.vy = vy
        self.r = r
        self.st = st
        self.T = T

class Results:
    def __init__(self, delta_st, delta_T, n, mu, vx, vy, r, st, T, s, s_dot, cost, optimizer_results):
        self.delta_st = delta_st
        self.delta_T = delta_T
        self.n = n
        self.mu = mu
        self.vx = vx
        self.vy = vy
        self.r = r
        self.st = st
        self.T = T
        self.s = s 
        self.s_dot = s_dot
        self.cost = cost
        self.optimizer_results = optimizer_results

class Optimizer:
    
    def __init__(self, Track=None, Model_Parameters=None, print_level=5):
        self.Track = Track
        self.modelParam = Model_Parameters
        self.results = None
        self.print_level = print_level
    
    def createTrack(self, Track):
        self.Track = Track

    def createModelParameter(self, Model_Parameters):
        self.modelParam = Model_Parameters

    # Define the functions that govern the states as constaints
    def calc_normal_dist_dot(self, m, n):
        if n > 1:
            return (m.normal_dist[n]*self.calc_centre_dist_dot(m, n-1)) ==  (m.normal_dist[n-1]*self.calc_centre_dist_dot(m, n-1)) +  ( (self.Track.delta_s) * ( (m.long_vel[n-1]*sin(m.local_heading[n-1])) + (m.lat_vel[n-1]*cos(m.local_heading[n-1]))) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip
    def calc_local_heading_dot(self, m, n):
        if n > 1:
            return (m.local_heading[n]*self.calc_centre_dist_dot(m, n-1)) == (m.local_heading[n-1]*self.calc_centre_dist_dot(m, n-1)) +  ( (self.Track.delta_s) * (m.yaw_rate[n-1] - (self.Track.curvature[n-1] * self.calc_centre_dist_dot(m, n-1))) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip 
    def calc_lat_vel_dot(self, m, n):
        if n > 1:
            return (m.lat_vel[n]*self.calc_centre_dist_dot(m, n-1)) == (m.lat_vel[n-1]*self.calc_centre_dist_dot(m, n-1)) +  ( (self.Track.delta_s) * (1/m.mass) * (self.calc_rear_LateralForce(m, (n-1)) + ((self.calc_front_LateralForce(m, (n-1)))*(cos(m.steer[n-1]))) - (m.mass*(m.long_vel[n-1])*(m.yaw_rate[n-1])) ) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip
    def calc_long_vel_dot(self, m, n):
        if n > 1:
            return (m.long_vel[n]*self.calc_centre_dist_dot(m, n-1)) == (m.long_vel[n-1]*self.calc_centre_dist_dot(m, n-1)) +  ( (self.Track.delta_s) * (1/m.mass) * (self.calc_F_long(m, (n-1)) - ((self.calc_front_LateralForce(m, (n-1)))*(sin(m.steer[n-1]))) + (m.mass*m.lat_vel[n-1]*m.yaw_rate[n-1])  ) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip
    def calc_yaw_rate_dot(self, m, n):
        if n > 1:
            return (m.yaw_rate[n]*self.calc_centre_dist_dot(m, n-1)) == (m.yaw_rate[n-1]*self.calc_centre_dist_dot(m, n-1)) +  ( (self.Track.delta_s) * (1/m.I_z) * ((self.calc_front_LateralForce(m, (n-1)) * m.L_forward * cos(m.steer[n-1])) - (self.calc_rear_LateralForce(m, (n-1)) * m.L_rear) + self.calc_torqueMoment(m, (n-1))) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip  
    def calc_steer_dot(self, m, n):
        if n > 1:
            return (m.steer[n]*self.calc_centre_dist_dot(m, n-1)) == (m.steer[n-1]*self.calc_centre_dist_dot(m, n-1)) +  ( (self.Track.delta_s) * (m.delta_steer[n-1]) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip           
    def driver_command_dot(self, m, n):
        if n > 1:
            return (m.driver_command[n]*self.calc_centre_dist_dot(m, n-1)) == (m.driver_command[n-1]*self.calc_centre_dist_dot(m, n-1)) +  ( (self.Track.delta_s) * (m.delta_driver_command[n-1]) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip
      
    # Define the functions that govern the additional variables
    def calc_rear_slip_angle(self, m, n):
        temp = ((m.lat_vel[n] - (m.L_rear * m.yaw_rate[n])) / (m.long_vel[n] + 0.00000001))
        alpha_r = -atan(temp)
        return alpha_r 
    def calc_rear_LateralForce(self, m, n):
        alpha_r = self.calc_rear_slip_angle(m, n)
        temp = m.C_rear * atan((m.B_rear * alpha_r))
        rear_lateral_Force = m.rear_NormalForce * m.D_rear * sin(temp)
        return rear_lateral_Force
    def calc_front_slip_angle(self, m, n):
        temp = ((m.lat_vel[n] + (m.L_forward * m.yaw_rate[n])) / (m.long_vel[n] + 0.00000001))
        alpha_f = (-atan(temp)  + m.steer[n])
        return alpha_f
    def calc_front_LateralForce(self, m, n):
        alpha_f = self.calc_front_slip_angle(m, n)
        temp = m.C_front * atan(m.B_front * alpha_f)
        front_lateral_Force = m.front_NormalForce * m.D_front * sin(temp)
        return front_lateral_Force
    def calc_YawRateTarget(self, m, n):
        temp = ((tan(m.steer[n])) * m.long_vel[n]) / (m.L_rear + m.L_forward)
        return temp
    def calc_torqueMoment(self, m, n):
        temp = m.P_tv * (self.calc_YawRateTarget(m, n) - m.yaw_rate[n])
        return temp
    def calc_F_long(self, m, n):
        temp = (m.C_m * m.driver_command[n]) - (m.C_r0) - (m.C_r2 * ((m.long_vel[n])**2))
        return temp
    def calc_centre_dist_dot(self, m, n):
        temp = ( ((m.long_vel[n] * cos(m.local_heading[n])) - (m.lat_vel[n] * sin(m.local_heading[n]))) / (1 - ((m.normal_dist[n])*(self.Track.curvature[n]))) )
        return temp

    # Define the path constraints
    def setLeft_width(self, m, n): 
        if n < self.Track.N:
             #return ( m.normal_dist[n] + ((m.L_c/2) * sin(abs(m.local_heading[n]))) + ((m.W_c/2) * cos(m.local_heading[n])) )<= self.Track.width[n] 
             return ( m.normal_dist[n] + ((m.L_c/2) * 1) + ((m.W_c/2) * cos(m.local_heading[n])) )<= self.Track.width[n] 
        else:
            return Constraint.Skip
    def setRight_width(self, m, n):
        if n < self.Track.N:
            #return ( -m.normal_dist[n] + ((m.L_c/2) * sin(abs(m.local_heading[n]))) + ((m.W_c/2) * cos(m.local_heading[n])) )<= self.Track.width[n] 
            return ( -m.normal_dist[n] + ((m.L_c/2) * 1) + ((m.W_c/2) * cos(m.local_heading[n])) )<= self.Track.width[n] 
        else:
            return Constraint.Skip
    
    # Define Obstacle Path constraints:
    def setAddition_Constraint(self, m, n):
        if self.Track.obstacles.start_index <= n and self.Track.obstacles.end_index > n:
            lower_bound = self.Track.obstacles.widths[0, n]
            upper_bound = self.Track.obstacles.widths[1, n]
            return m.normal_dist[n] == ( (m.normal_distance_minus[n]- lower_bound) + (m.normal_distance_plus[n] - upper_bound))
        else:
            return Constraint.Skip
    def setComplementary_Bound(self, m, n):
        if self.Track.obstacles.start_index <= n and self.Track.obstacles.end_index > n:
            lower_bound = self.Track.obstacles.widths[0, n]
            upper_bound = self.Track.obstacles.widths[1, n]
            return (m.normal_distance_minus[n] - lower_bound) * (m.normal_distance_plus[n] - upper_bound) == 0
        else:
            return Constraint.Skip
    def set_exclusion_bound_minus(self, m, n):
        if self.Track.obstacles.start_index <= n and self.Track.obstacles.end_index > n:
            lower_bound = self.Track.obstacles.widths[0, n]
            return m.normal_distance_minus[n] <= lower_bound
        else:
            return Constraint.Skip
    def set_exclusion_bound_plus(self, m, n):
        if self.Track.obstacles.start_index <= n and self.Track.obstacles.end_index > n:
            upper_bound = self.Track.obstacles.widths[1, n]
            return m.normal_distance_plus [n] >= upper_bound
        else:
            return Constraint.Skip
        
    # Define Friction Constraints
    def setRear_Fric(self, m, n):
        if n < self.Track.N:
            return (((m.P_long * ((m.C_m * m.driver_command[n])/2))**2) + self.calc_rear_LateralForce(m, n) ) <= ((m.phi * m.D_rear)**2)   
        else:
            return Constraint.Skip
    def setFront_Fric(self, m, n):
        if n < self.Track.N:
            return (((m.P_long * ((m.C_m * m.driver_command[n])/2))**2) + self.calc_front_LateralForce(m, n) ) <= ((m.phi * m.D_front)**2)   
        else:
            return Constraint.Skip

    # Define constraint that forces prgression along the track
    def progressForward(self, m, n):
        if n < self.Track.N:
            return self.calc_centre_dist_dot(m, n) >= 0
        else:
            return Constraint.Skip

    # Define the Objective Function
    # Additional Function: Slip angle fucntion
    def regularize_slipAngle(self, m, n):
        kinetic_slip = atan((m.steer[n] * m.L_rear)/(m.L_rear + m.L_forward))
        dynamic_slip = atan((m.lat_vel[n])/(m.long_vel[n]))
        return (m.q_B * ((dynamic_slip - kinetic_slip)**2))
    # Additional Function: Regularization of the inputs
    def regularize_Input(self, m, n):
        reg_1_steer = ((m.delta_steer[n]**2) * m.reg_delta_steer)
        reg_2_driverCommand = ((m.delta_driver_command[n]**2) * m.reg_delta_T)
        return (reg_1_steer + reg_2_driverCommand)
    def CostFunction(self, m):
        c = sum( ((self.Track.delta_s / (self.calc_centre_dist_dot(m, j))) + self.regularize_slipAngle(m, j) + self.regularize_Input(m, j)) for j in range(1, self.Track.N) )
        #c = sum( ((self.Track.delta_s / (self.calc_centre_dist_dot(m, j)))) for j in range(1, self.Track.N) )
        return c

    # Additional Functions for processing the result
    def calc_centre_dist_dot_value(self, m, n):
        temp = ((m.long_vel[n]() * cos(m.local_heading[n]())) - (m.lat_vel[n]() * sin(m.local_heading[n]()))) / (1 - ((m.normal_dist[n]())*(self.Track.curvature[n])))
        return temp
    def regularize_slipAngle_value(self, m, n):
        kinetic_slip = atan((m.steer[n]() * m.L_rear)/(m.L_rear + m.L_forward))
        dynamic_slip = atan((m.lat_vel[n]())/(m.long_vel[n]()))
        return (m.q_B * ((dynamic_slip - kinetic_slip)**2))
    def regularize_Input_value(self, m, n):
        reg_1_steer = ((m.delta_steer[n]()**2) * m.reg_delta_steer)
        reg_2_driverCommand = ((m.delta_driver_command[n]()**2) * m.reg_delta_T)
        return (reg_1_steer + reg_2_driverCommand)
    
    def myCostFunction(self, m):
        c = sum( ((self.Track.delta_s / (self.calc_centre_dist_dot_value(m, j))) + self.regularize_slipAngle_value(m, j) + self.regularize_Input_value(m, j)) for j in range(1, self.Track.N) )
        return c
    
    def export_Results(self, results, track, modelParameters, n_plus=None, n_minus=None, save_path=None):
            path = ""
            if save_path is None:
                if track.Name is not None:
                    if type(track.Name) == str:
                        path = "./Optimization of " + str(track.Name)
                    else:
                        path = "./Optimization Attempt with No Name Track"
                else:

                    path = "./Optimization Attempt with No Name Track"
            else:
                if track.Name is not None:
                    if type(track.Name) == str:
                        path = save_path + "/Optimization of " + str(track.Name)
                    else:
                        path = save_path + "/Optimization Attempt with No Name Track"
                else:

                    path = save_path + "/Optimization Attempt with No Name Track"
            try:
                #print("trying to make dictionary")
                os.makedirs(path , exist_ok=True)
                #print("directory made")
            except: Exception("Failed to make path")

            #print("Starting to make the CSV")
            csv_path = path + "/results.csv"
            df = pd.DataFrame({"delta_st" : results.delta_st, "delta_t" : results.delta_T, "normal_dist" : results.n, "local_heading" : results.mu, "x_vel" : results.vx, "y_vel" : results.vy, "yaw_rate" : results.r, "steering_angle" : results.st, "driver_command" : results.T})
            df.to_csv(csv_path, index=False)

            plt.plot(results.s[0:(track.N-1)], results.s_dot)
            plt.xlabel('S')
            plt.ylabel('S_Dot')
            fig_path = path + "/s_dot.png"
            plt.savefig(fig_path)
            plt.close()
      
            plt.plot(results.s)
            plt.xlabel('Index')
            plt.ylabel('S')
            fig_path = path + "/s.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(results.s, results.delta_st)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Change in Steering angle: Driver Input')
            fig_path = path + "/delta_st.png"
            plt.savefig(fig_path)
            plt.close()
       
            plt.plot(results.s, results.delta_T)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Change in Steering angle: Driver Input')
            fig_path = path + "/delta_T.png"
            plt.savefig(fig_path)
            plt.close()

            plt.close()
            plt.plot(results.s, results.n)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Orthognal distance from the centre line')
            fig_path = path + "/n.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(results.s, results.mu)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Local heading')
            fig_path = path + "/mu.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(results.s, results.vx)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Longitutinal Velocity')
            fig_path = path + "/vx.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(results.s, results.vy)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Latitudinal Velocity')
            fig_path = path + "/vy.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(results.s, results.r)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Yaw Rate')
            fig_path = path + "/r.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(results.s, results.st)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Steering angle')
            fig_path = path + "/steer.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(results.s, results.T)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Driver Command')
            fig_path = path + "/T.png"
            plt.savefig(fig_path)
            plt.close()
            
            if n_minus is not None and n_plus is not None:
                plt.plot(results.s, n_plus)
                plt.xlabel('Distance along the centre line')
                plt.ylabel('n_plus')
                fig_path = path + "/n_plus.png"
                plt.savefig(fig_path)
                plt.close()

                plt.plot(results.s, n_minus)
                plt.xlabel('Distance along the centre line')
                plt.ylabel('n_minus')
                fig_path = path + "/n_minus.png"
                plt.savefig(fig_path)
                plt.close()
            #print("CSV and pictures made")

            text_path = path + "/Results.txt"
            f = open(text_path, 'w')
            if track.Name is not None:
                if type(track.Name) == str:
                    f.write("Optimizing Track with Track Name: " + track.Name + "\n") 
                else: 
                    f.write("Optimizing Track with Track Name: No Name\n") 
            else: 
                f.write("Optimizing Track with Track Name: No Name\n")
            f.write("Final Cost Function: " + str(results.cost) + "\n")
            f.write(results.optimizer_results.solver.status + "\n") # tells you if the solver had any errors/ warnings
            f.write(results.optimizer_results.solver.termination_condition + "\n")
            f.close()
            track.exportTrackParameters(path)
            modelParameters.exportModelParameters(path)
            #print("END of function")
            return

    def test1(self, m, n):
        return ((m.long_vel[n]() * cos(m.local_heading[n]())) - (m.lat_vel[n]() * sin(m.local_heading[n]())))
    def test2(self, m, n): 
        return (1 - ((m.normal_dist[n]())*(self.Track.curvature[n])))

    def optimize(self, initCondition, Parameters=None, finalCondition=None, Temp_Track=None, export_results=False, save_path=None):
        if Parameters is not None:
            param = Parameters
        elif self.modelParam is not None:
            param = self.modelParam
        else:
            raise Exception("No Model Parameters")
        hold_track = copy.deepcopy(self.Track)
        if Temp_Track is not None:
            self.Track = Temp_Track
        elif self.Track is None:
            raise Exception("No Associated Track")
        
        # create the model
        m = ConcreteModel()

        # Define Used sets for the model
        m.N = RangeSet(self.Track.N) # Number of discretizations

        # Define Parameters of the model
        m.g = Param(initialize = param.g) # Gravitional Acceleration
        m.L_c = Param(initialize = param.L_c) # Total length of the car
        m.W_c = Param(initialize = param.W_c) # Width of the Car
        m.L_rear = Param(initialize = param.L_rear) # Distance from the center of gravity to the back of the car
        m.L_forward = Param(initialize = param.L_forward) # Distance from the center of gravity to the front of the car
        m.mass = Param(initialize = param.mass) # Mass of the car
        m.I_z = Param(initialize = param.I_z) # Rotational Inertia of the car
        m.P_tv = Param(initialize = param.P_tv) # Gain of the System for the moment the torque vectoring system produces
        m.C_m = Param(initialize = param.C_m) # Motor Gain
        m.C_r0 = Param(initialize = param.C_r0) # Rolling Resistance
        m.C_r2 = Param(initialize = param.C_r2) # Drag Resistance
        m.P_long = Param(initialize = param.P_long) #4# Shape of the friction Ellipse
        m.phi = Param(initialize = param.phi) # Maximum Combined Lateral Force
        m.q_B = Param(initialize = param.q_B) # Slip Angle Regularization Weight
        m.reg_delta_steer = Param(initialize = param.reg_delta_steer) # Regularization term for the change in steering input
        m.reg_delta_T = Param(initialize = param.reg_delta_T) # Regularization term for the change in driver command input
        # Tire Model Parameters
        m.B_rear = Param(initialize = param.rearTyre.B)
        m.C_rear = Param(initialize = param.rearTyre.C)
        m.D_rear = Param(initialize = param.rearTyre.D)
    
        m.B_front = Param(initialize = param.frontTyre.B)
        m.C_front = Param(initialize = param.frontTyre.C)
        m.D_front = Param(initialize = param.frontTyre.D)

        m.rear_NormalForce = Param(initialize = ((m.L_rear * m.mass * m.g) / (m.L_forward + m.L_rear)))
        m.front_NormalForce = Param(initialize = ((m.L_forward * m.mass * m.g) / (m.L_forward + m.L_rear)))

        # BOUNDS
        # Minimum possible steering angle
        min_steer = -np.pi/3
        # Maximum possible steering angle
        max_steer = np.pi/3
        # Minimum possible change in steering angle at any given point
        min_delta_steer = -np.pi/6
        # Maximum possible change in steering angle at any given point
        max_delta_steer = np.pi/6
        # Minimum possible Driver Command
        min_driver_command  = -1
        # Maximum possible Driver Command
        max_driver_command = 1
        # Minimum possible change in Driver Command at any given point
        min_delta_driver_command = -0.2
        # Maximum possible change in Driver Command at any given point
        max_delta_driver_command = 0.2

        # Define Control Inputs
        m.delta_steer = Var(m.N, bounds = (min_steer, max_steer))
        m.delta_driver_command = Var(m.N)

        # Define the dependent variables (State Variables)
        m.normal_dist = Var(m.N)
        m.local_heading = Var(m.N, bounds = ((-np.pi/2), (np.pi/2)))
        m.long_vel = Var(m.N)
        m.lat_vel = Var(m.N)
        m.yaw_rate = Var(m.N)
        m.steer = Var(m.N, bounds = (min_steer, max_steer))
        m.driver_command = Var(m.N, bounds = (min_driver_command, max_driver_command))

        # Define the functions that govern the states as constaints
        m.next_state_n = Constraint(m.N, rule = self.calc_normal_dist_dot)
        m.next_state_u = Constraint(m.N, rule = self.calc_local_heading_dot)
        m.next_state_vx = Constraint(m.N, rule = self.calc_long_vel_dot)
        m.next_state_vy = Constraint(m.N, rule = self.calc_lat_vel_dot)
        m.next_state_r = Constraint(m.N, rule = self.calc_yaw_rate_dot)
        m.next_state_s_dot = Constraint(m.N, rule = self.calc_steer_dot)
        m.next_state_T_dot = Constraint(m.N, rule = self.driver_command_dot)

        # # Define the path constraints
        m.setT_Left = Constraint(m.N, rule=self.setLeft_width)
        m.setT_Right = Constraint(m.N, rule=self.setRight_width)

        # Define the constraints for the path if there is an obstacle
        if self.Track.obstacles is not None:
            print("here")
            m.normal_distance_plus = Var(m.N)
            m.normal_distance_minus = Var(m.N)
            m.con1 = Constraint(m.N, rule = self.setAddition_Constraint)
            m.con2 = Constraint(m.N, rule = self.setComplementary_Bound)
            m.con3 = Constraint(m.N, rule = self.set_exclusion_bound_minus)
            m.con4 = Constraint(m.N, rule = self.set_exclusion_bound_plus)
            pass
        
        # # Define Friction Constraints
        m.setR_Fric = Constraint(m.N, rule=self.setRear_Fric)
        m.setF_Fric = Constraint(m.N, rule=self.setFront_Fric)

        # # Define constraint that forces prgression along the track
        m.progressForward = Constraint(m.N, rule=self.progressForward)

        # Define the Objective Function
        m.Cost = Objective(rule=self.CostFunction)

        # Initialization
        for n in range(1,(self.Track.N)+1):
            # m.delta_steer[n].value = 0#(np.random.random()*(np.pi/3)) - np.pi/6
            # m.delta_driver_command[n].value = 0#(np.random.random()*(2)) - 1
            # m.normal_dist[n].value = 0#(np.random.random()*(np.amax(self.Track.width)*2)) - np.amax(self.Track.width)
            # m.local_heading[n].value = 0#(np.random.random()*(np.pi)) - (np.pi/2)
            m.long_vel[n].value = initCondition.vx#np.random()
            # m.lat_vel[n].value = 0#(np.random.random()*(2)) - 1
            # m.yaw_rate[n].value = 0#(np.random.random()*np.pi/3) - np.pi/6
            # m.steer[n].value = 0#(np.random.random()*(np.pi/3)) - np.pi/6
            # m.driver_command[n].value = 0#(np.random.random()*(2)) - 1


        # Initial Conditions at n = 1
        m.delta_steer[1].value = initCondition.delta_st
        m.delta_steer[1].fixed = True
        ###
        m.delta_driver_command[1].value =  initCondition.delta_T
        m.delta_driver_command[1].fixed =  True
        ###
        m.normal_dist[1].value = initCondition.n
        m.normal_dist[1].fixed = True
        ###
        m.local_heading[1].value = initCondition.mu
        m.local_heading[1].fixed = True
        ###
        m.long_vel[1].value = initCondition.vx
        m.long_vel[1].fixed = True
        ###
        m.lat_vel[1].value = initCondition.vy 
        m.lat_vel[1].fixed = True
        ###
        m.yaw_rate[1].value = initCondition.r 
        m.yaw_rate[1].fixed = True
        ###
        m.steer[1].value = initCondition.st
        m.steer[1].fixed = True
        ###
        m.driver_command[1].value = initCondition.T
        m.driver_command[1].fixed = True

        if finalCondition is not None:
            # Final Conditions at n = N
            m.delta_steer[self.Track.N].value = finalCondition.delta_st
            m.delta_steer[self.Track.N].fixed = True
            ###
            m.delta_driver_command[self.Track.N].value =  finalCondition.delta_T
            m.delta_driver_command[self.Track.N].fixed =  True
            ###
            m.normal_dist[self.Track.N].value = finalCondition.n
            m.normal_dist[self.Track.N].fixed = True
            ###
            m.local_heading[self.Track.N].value = finalCondition.mu
            m.local_heading[self.Track.N].fixed = True
            ###
            m.long_vel[self.Track.N].value = finalCondition.vx
            m.long_vel[self.Track.N].fixed = True
            ###
            m.lat_vel[self.Track.N].value = finalCondition.vy 
            m.lat_vel[self.Track.N].fixed = True
            ###
            m.yaw_rate[self.Track.N].value = finalCondition.r 
            m.yaw_rate[self.Track.N].fixed = True
            ###
            m.steer[self.Track.N].value = finalCondition.st
            m.steer[self.Track.N].fixed = True
            ###
            m.driver_command[self.Track.N].value = finalCondition.T
            m.driver_command[self.Track.N].fixed = True


        # Solver
        #opt = SolverFactory('ipopt',executable = r"/root/home/oran/Downloads/2022Masters-master/Ipopt_3.14.5/bin/ipopt.exe") 
        opt = SolverFactory('ipopt',executable = r"~/Downloads/ipopt-linux64/ipopt") 
        #opt = SolverFactory('ipopt')
        opt.options["print_level"] = self.print_level # prints a log with each iteration (you want to this - it's the only way to see progress.)
        opt.options["max_iter"] = 30000000 # maximum number of iterations
        opt.options["max_cpu_time"] = (3600*2) # maximum cpu time in seconds
        opt.options["Tol"] = 1e-6 # the tolerance for feasibility. Considers constraints satisfied when they're within this margin.   
        opt.options["halt_on_ampl_error"] = 'yes'

        try:
            results = opt.solve(m, tee = True, symbolic_solver_labels=True)
                  
        except Exception:

            # if export_results:
            #     if self.Track.Name is not None:
            #         if type(self.Track.Name) == str:
            #             path = "./" + self.Track.Name
            #         else:
            #             path = "./Optimization Attempt with No Name Track"
            #     else:
            #         path = "./Optimization Attempt with No Name Track"
            #     os.makedirs(path , exist_ok=True)
            #     temp = np.zeros(1)
            #     csv_path = path + "/FAILED.csv"
            #     df = pd.DataFrame({"Attempt" : temp})
            #     df.to_csv(csv_path, index=False)
                
            self.Track = copy.deepcopy(hold_track)
            return None

        else: 
            # Process Results
            path = "./"
            s = np.array([n*self.Track.delta_s for n in m.N])
            s_dot = np.array([self.calc_centre_dist_dot_value(m, N) for N in range(1, self.Track.N)])
            delta_st = np.array([m.delta_steer[N]() for N in m.delta_steer])
            delta_T = np.array([m.delta_driver_command[N]() for N in m.delta_driver_command])

            n = np.array([m.normal_dist[N]() for N in m.normal_dist])
            local_heading = np.array([m.local_heading[N]() for N in m.local_heading])
            x_vel = np.array([m.long_vel[N]() for N in m.long_vel])
            y_vel = np.array([m.lat_vel[N]() for N in m.lat_vel])
            yaw_rate = np.array([m.yaw_rate[N]() for N in m.yaw_rate])
            st = np.array([m.steer[N]() for N in m.steer])
            T = np.array([m.driver_command[N]() for N in m.driver_command])
            cost = self.myCostFunction(m)
            res = Results(delta_st, delta_T, n, local_heading, x_vel, y_vel, yaw_rate, st, T, s, s_dot, cost, results)
            self.results = res
            
        
            # top_term = np.array([self.test1(m, N) for N in range(1, self.Track.N)])
            # bottom_term = np.array([self.test2(m, N) for N in range(1, self.Track.N)])
            
            # fig, axs = plt.subplots(2)
            # axs[0].plot(top_term)
            # axs[0].set_title('Top term')
            # axs[1].plot(bottom_term)
            # axs[1].set_title('Bottom term')
            # plt.show()
            if self.Track.obstacles is not None:
                n_minus = np.array([m.normal_distance_minus[N]() for N in m.normal_distance_minus])
                n_plus = np.array([m.normal_distance_plus[N]() for N in m.normal_distance_plus])
                print(n_plus)
            else: 
                n_minus = None
                n_plus = None

            if export_results:
                self.export_Results(self.results, self.Track, param, n_minus=n_minus, n_plus=n_plus, save_path=save_path)

            self.Track = copy.deepcopy(hold_track)
            return res


if __name__ == "__main__":
    # delta_st = 0.0003629213412060902
    # delta_T = 2.580279508249128e-05
    # n = -0.9395411847116497
    # mu = -0.026399454570116852
    # vx = 3.525649458586957
    # vy = -0.02898014799084816
    # r = -0.9152654039443445
    # st = -0.012748016254967602
    # T = -5.435195497131963e-11

   
    # m = map('columbia_small') #Works
    # single_track = m.generate_track(0.1)

    # #initCondition = Kinematic_Bicycle_Model_States(delta_st, delta_T, n, mu, vx, vy, r, st, T)
    # initCondition = Kinematic_Bicycle_Model_States(0, 0, 0, 0, 0.5, 0, 0, 0, 0)
    # param = Kinematic_Bicycle_Model_Parameters(Slip_AngleReg=5, Steer_Reg=5, Max_Tyre_Force=5.0)
    # # track_twice = np.zeros(len(single_track.curvature)*2)
    # # for i in range(len(single_track.curvature)):
    # #     track_twice[i] = single_track.curvature[i]
    # # for i in range(len(single_track.curvature), len(track_twice)):
    # #     track_twice[i] = single_track.curvature[i-len(single_track.curvature)]
    # # track = Track(len(track_twice), 0.1, track_twice, 1.4)
    # opt = Optimizer(single_track, param, print_level=5)
    # initial_plan = opt.optimize(initCondition, Parameters=param, Temp_Track=single_track)
    
    # second_opt = Optimizer(single_track, param, print_level=5)
    # N = len(initial_plan.vx)-4

    # second_init_conditions = Kinematic_Bicycle_Model_States(initial_plan.delta_st[N-1], initial_plan.delta_T[N-1], initial_plan.n[N], initial_plan.mu[N], initial_plan.vx[N], initial_plan.vy[N], initial_plan.r[N], initial_plan.st[N], initial_plan.T[N])
    # # second_init_conditions = Kinematic_Bicycle_Model_States(0, 0, initial_plan.n[N], initial_plan.mu[N], initial_plan.vx[N], initial_plan.vy[N], initial_plan.r[N], initial_plan.st[N], initial_plan.T[N])
    # final_plan = second_opt.optimize(second_init_conditions, save_path="~/Downloads/Second_Attempt", export_results=False)
    
    N = 300 
    d_s = 0.1
    curvature = np.zeros(N)
    width = 1.4
    obsacle_width = np.zeros((2, N))
    obs_strt_ind = 200
    obs_end_ind = 210
    for i in range(obs_strt_ind, obs_end_ind):
        obsacle_width[0, i] = 0.5
        obsacle_width[1, i] = 0.6
    obstacle = Obstacle(2, 0.1, (obs_strt_ind*0.1), obs_strt_ind, (obs_end_ind*0.1), obs_end_ind, obsacle_width)

    track = Track(N, d_s, curvature, width, Obstacles=obstacle)
    init_conditions = Kinematic_Bicycle_Model_States(0.0, 0.0, 0.55, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    param =  Kinematic_Bicycle_Model_Parameters(Slip_AngleReg=5.0, Steer_Reg=5.0, Max_Tyre_Force=15.0, Motor_gain=8.0, Friction_Ellipse=2.0)
    opt = Optimizer(track, param, print_level=5)
    plan = opt.optimize(init_conditions, Parameters=param, Temp_Track=track, export_results=True, save_path=r"./OBSTACLE")