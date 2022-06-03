# import libraries
# Pyomo stuff
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
from IPython.display import display #for pretty printing

import numpy as np
import matplotlib.pyplot as plt
import copy




class Track:
    def __init__(self, N, delta_S, Curvature=None, Width=None, Name=None):
        if (type(N) == int):
            self.N = N
        else:
            raise Exception("Incorrect argument for N")  
        
        if  (type(delta_S) == int or type(delta_S) == float):
            self.delta_s = delta_S
        else:
            raise Exception("Incorrect argument for Delta S")

        if Name is not None: 
            if type(Name) == str:
                self.Name = Name
            else:
                raise Exception("Incorrect argument for Name")
        else:
            self.Name = None

        if (Curvature is not None) and (Width is not None):
            self.new_track(Curvature, Width, Track_Length=self.N)


    def set_constant_curvature(self, Curvature, Start=None, Stop=None):
        if (type(Curvature) == int or type(Curvature) == float):
            raise Exception("Incompatible Arguments")
        if Start is None:
            for i in range(self.N):
                self.curvature = Curvature
        elif Stop is None:
            if (type(Start) != int):
                raise Exception("Incompatible Argument for Start")
            for i in range(Start, self.N):
                self.curvature = Curvature
        else:
            if (type(Start) != int) or (type(Stop) != int) :
                raise Exception("Incompatible Argument for Start")
            for i in range(Start, Stop):
                self.curvature = Curvature

    def set_constant_width(self, Width, Start=None, Stop=None):
        if (type(Width) == int or type(Width) == float):
            raise Exception("Incompatible Arguments")
        if Start is None:
            for i in range(self.N):
                self.width = Width
        elif Stop is None:
            if (type(Start) != int):
                raise Exception("Incompatible Argument for Start")
            for i in range(Start, self.N):
                self.width = Width
        else:
            if (type(Start) != int) or (type(Stop) != int) :
                raise Exception("Incompatible Argument for Start")
            for i in range(Start, Stop):
                self.width = Width

    def new_track(self, Curvature, Width, Track_Length=None):
        if not ((hasattr(Curvature, '__len__') or (type(Curvature) == int or type(Curvature) == float)) and (hasattr(Width, '__len__') or (type(Width) == int or type(Width) == float))):
            raise Exception("Incompatible Arguments Given for Cuvature and Widths")
        
        if Track_Length is None:
            if (type(Curvature) == int or type(Curvature) == float) and (type(Width) == int or type(Width) == float):
                raise Exception("No Track length Given")
            
            if hasattr(Curvature, '__len__') and hasattr(Width, '__len__'):
                if len(Curvature) != len(Width):
                    raise Exception("Inconsistent Track length between Curvature and Width")
                self.N = len(Curvature)
                self.curvature = np.zeros(self.N)
                self.width = np.zeros(self.N)
                for i in range(self.N):
                    self.curvature[i] = Curvature[i]
                    self.width[i] = Width[i]
            
            elif hasattr(Curvature, '__len__') and (type(Width) == int or type(Width) == float):
                self.N = len(Curvature)
                self.curvature = np.zeros(self.N)
                self.width = np.zeros(self.N)
                for i in range(self.N):
                    self.curvature[i] = Curvature[i]
                    self.width[i] = Width
            
            elif  (type(Curvature) == int or type(Curvature) == float) and hasattr(Width, '__len__'):
                self.N = len(Width)
                self.curvature = np.zeros(self.N)
                self.width = np.zeros(self.N)
                for i in range(self.N):
                    self.curvature[i] = Curvature
                    self.width[i] = Width[i]
            else:
                raise Exception("Incompatible Argument Types")
     
        if Track_Length is not None:
            if (type(Track_Length) == int):
                if hasattr(Curvature, '__len__'):
                    if len(Curvature) != Track_Length:
                        raise Exception("Inconsistent Track length between Curvature and Track_Length")
                if hasattr(Width, '__len__'):
                    if len(Width) != Track_Length:
                        raise Exception("Inconsistent Track length between Width and Track_Length")
                if (type(Curvature) == int or type(Curvature) == float) and (type(Width) == int or type(Width) == float):
                    self.N = Track_Length
                    self.curvature = np.zeros(self.N)
                    self.width = np.zeros(self.N)
                    for i in range(self.N):
                        self.curvature[i] = Curvature
                        self.width[i] = Width

                elif hasattr(Curvature, '__len__') and hasattr(Width, '__len__'):
                    if len(Curvature) != len(Width):
                        raise Exception("Inconsistent Track length between Curvature and Width")
                    self.N = Track_Length
                    self.curvature = np.zeros(self.N)
                    self.width = np.zeros(self.N)
                    for i in range(self.N):
                        self.curvature[i] = Curvature[i]
                        self.width[i] = Width[i]
            
                elif hasattr(Curvature, '__len__') and (type(Width) == int or type(Width) == float):
                    self.N = Track_Length
                    self.curvature = np.zeros(self.N)
                    self.width = np.zeros(self.N)
                    for i in range(self.N):
                        self.curvature[i] = Curvature[i]
                        self.width[i] = Width
            
                elif  (type(Curvature) == int or type(Curvature) == float) and hasattr(Width, '__len__'):
                    self.N = Track_Length
                    self.curvature = np.zeros(self.N)
                    self.width = np.zeros(self.N)
                    for i in range(self.N):
                        self.curvature[i] = Curvature
                        self.width[i] = Width[i]
                else:
                    raise Exception("Incompatible Argument Types")
            else:
                raise Exception("Incorrect format for Track_Length")
    
    def exportTrackParameters(self, path):
        line1 = "Number of Track discretizations: " + str(self.N) + "\n"
        line2 = "Discretization Distance: " + str(self.delta_s) + "\n"
        if self.Name is not None:
            if type(self.Name) == str:
                line3 = "Track name: " + str(self.Name) + "\n"
        else: 
            line3 = "Track name: No Name Given\n"
        temp_path = path + "/Track Information.txt" 
        f = open(temp_path, "w")
        f.write(line1)
        f.write(line2)
        f.write(line3)
        f.close()
        df = pd.DataFrame({"Curvature" : self.curvature, "Width" : self.width})
        csv_path = path + "/Track Information.csv" 
        df.to_csv(csv_path, index=False)
        
class Tyre_Parameters:
    def __init__(self, B=10.0, C=1.9, D=1.0):
        self.B = B
        self.C = C
        self.D = D

class State_parameters:
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

class Model_Parameters:
    def __init__(self, Car_Length=0.58, Car_Width=0.31, COF_R_Axle_len=0.17145, COF_F_Axle_len=0.15875, mass=10.0, RotateZ_Inertia=0.04712, Torque_gain=0.0, Motor_gain=5.0, Roll_Resist=0.0, Drag_Resist=0.2, Friction_Ellipse=4.0, Max_Tyre_Force=500.0, Slip_AngleReg=0.2, Steer_Reg=0.1, Drive_Reg=0.0, rearTyre=None, frontTyre=None):
        self.g = 9.81 # Gravitional Acceleration
        self.L_c = Car_Length # Total length of the car
        self.W_c = Car_Width # Width of the Car
        self.L_rear = COF_R_Axle_len # Distance from the center of gravity to the back of the car
        self.L_forward = COF_F_Axle_len # Distance from the center of gravity to the front of the car
        self.mass = mass # Mass of the car
        self.I_z = RotateZ_Inertia # Rotational Inertia of the car
        self.P_tv = Torque_gain # Gain of the System for the moment the torque vectoring system produces
        self.C_m = Motor_gain # Motor Gain
        self.C_r0 = Roll_Resist # Rolling Resistance
        self.C_r2 = Drag_Resist# Drag Resistance
        self.P_long = Friction_Ellipse # Shape of the friction Ellipse
        self.phi = Max_Tyre_Force # Maximum Combined Lateral Force
        self.q_B = Slip_AngleReg # Slip Angle Regularization Weight
        self.reg_delta_steer = Steer_Reg # Regularization term for the change in steering input
        self.reg_delta_T = Drive_Reg # Regularization term for the change in driver command input

        if rearTyre is not None:
            self.rearTyre = rearTyre
        else:
            self.rearTyre = Tyre_Parameters()

        if frontTyre is not None:
            self.frontTyre = frontTyre
        else:
            self.frontTyre = Tyre_Parameters()
    
    def exportModelParameters(self, path):
        line1 = "Car Length: " + str(self.L_c) + "\n"
        line2 = "Car Width: " + str(self.W_c) + "\n"
        line3 = "Distance from the center of gravity to rear axle: " + str(self.L_rear) + "\n"
        line4 = "Distance from the center of gravity to front axle: " + str(self.L_forward) + "\n"
        line5 = "Car Mass: " + str(self.mass) + "\n"
        line6 = "Rotational Inertia of the car: " + str(self.I_z) + "\n"
        line7 = "Gain of the System for the moment the torque vectoring system produces: " + str(self.P_tv) + "\n"
        line8 = "Motor Gain: " + str(self.C_m) + "\n"
        line9 = "Rolling Resistance: " + str(self.C_r0) + "\n"
        line10 = "Drag Resistance: " + str(self.C_r2) + "\n"
        line11 = "Shape of the friction Ellipse: " + str(self.P_long) + "\n"
        line12 = "Maximum Combined Lateral Force: " + str(self.phi) + "\n"
        line13 = "Slip Angle Regularization Weight: " + str(self.q_B) + "\n"
        line14 = "Regularization term for the change in steering input: " + str(self.reg_delta_steer) + "\n"
        line15 = "Regularization term for the change in driver command input: " + str(self.reg_delta_T) + "\n"

        line16 = "Rear Tyre B Parameter: " + str(self.rearTyre.B) + "\n"
        line17 = "Rear Tyre C Parameter: " + str(self.rearTyre.C) + "\n"
        line18 = "Rear Tyre D Parameter: " + str(self.rearTyre.D) + "\n"

        line19 = "Front Tyre B Parameter: " + str(self.frontTyre.B) + "\n"
        line20 = "Front Tyre C Parameter: " + str(self.frontTyre.C) + "\n"
        line21 = "Front Tyre D Parameter: " + str(self.frontTyre.D) + "\n"
        temp_path = path + "/Model Parameters.txt"
        f = open(temp_path, "w")
        f.write(line1)
        f.write(line2)
        f.write(line3)
        f.write(line4)
        f.write(line5)
        f.write(line6)
        f.write(line7)
        f.write(line8)
        f.write(line9)
        f.write(line10)
        f.write(line11)
        f.write(line12)
        f.write(line13)
        f.write(line14)
        f.write(line15)
        f.write(line16)
        f.write(line17)
        f.write(line18)
        f.write(line19)
        f.write(line20)
        f.write(line21)
        f.close()

class Results:
    def __init__(self, delta_st, delta_T, n, mu, vx, vy, r, st, T, cost):
        self.delta_st = delta_st
        self.delta_T = delta_T
        self.n = n
        self.mu = mu
        self.vx = vx
        self.vy = vy
        self.r = r
        self.st = st
        self.T = T
        self.cost = cost

class Optimizer:
    
    def __init__(self, Track=None, Model_Parameters=None):
        self.Track = Track
        self.modelParam = Model_Parameters
        self.results = None
    
    def createTrack(self, Track):
        self.Track = Track

    def createModelParameter(self, Model_Parameters):
        self.modelParam = Model_Parameters

    # Define the functions that govern the states as constaints
    def calc_normal_dist_dot(self, m, n):
        if n > 1:
            return m.normal_dist[n] ==  m.normal_dist[n-1] +  ( (self.Track.delta_s/self.calc_centre_dist_dot(m, n-1)) * ( (m.long_vel[n-1]*sin(m.local_heading[n-1])) + (m.lat_vel[n-1]*cos(m.local_heading[n-1]))) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip
    def calc_local_heading_dot(self, m, n):
        if n > 1:
            return m.local_heading[n] == m.local_heading[n-1] +  ( (self.Track.delta_s/self.calc_centre_dist_dot(m, n-1)) * (m.yaw_rate[n-1] - (self.Track.curvature[n-1] * self.calc_centre_dist_dot(m, n-1))) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip 
 
        if n > 1:
            return m.long_vel[n] == m.long_vel[n-1] +  ( (self.Track.delta_s/self.calc_centre_dist_dot(m, n-1)) * (1/m.mass) * (self.calc_F_long(m, (n-1)) - ((self.calc_front_LateralForce(m, (n-1)))*(sin(m.steer[n-1]))) + (m.mass*m.lat_vel[n-1]*m.yaw_rate[n-1])  ) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip
    def calc_lat_vel_dot(self, m, n):
        if n > 1:
            return m.lat_vel[n] == m.lat_vel[n-1] +  ( (self.Track.delta_s/self.calc_centre_dist_dot(m, n-1)) * (1/m.mass) * (self.calc_rear_LateralForce(m, (n-1)) + ((self.calc_front_LateralForce(m, (n-1)))*(cos(m.steer[n-1]))) - (m.mass*(m.long_vel[n-1])*(m.yaw_rate[n-1])) ) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip
    def calc_long_vel_dot(self, m, n):
        if n > 1:
            return m.long_vel[n] == m.long_vel[n-1] +  ( (self.Track.delta_s/self.calc_centre_dist_dot(m, n-1)) * (1/m.mass) * (self.calc_F_long(m, (n-1)) - ((self.calc_front_LateralForce(m, (n-1)))*(sin(m.steer[n-1]))) + (m.mass*m.lat_vel[n-1]*m.yaw_rate[n-1])  ) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip
    def calc_yaw_rate_dot(self, m, n):
        if n > 1:
            return m.yaw_rate[n] == m.yaw_rate[n-1] +  ( (self.Track.delta_s/self.calc_centre_dist_dot(m, n-1)) * (1/m.I_z) * ((self.calc_front_LateralForce(m, (n-1)) * m.L_forward * cos(m.steer[n-1])) - (self.calc_rear_LateralForce(m, (n-1)) * m.L_rear) + self.calc_torqueMoment(m, (n-1))) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip  
    def calc_steer_dot(self, m, n):
        if n > 1:
            return m.steer[n] == m.steer[n-1] +  ( (self.Track.delta_s/self.calc_centre_dist_dot(m, n-1)) * (m.delta_steer[n-1]) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip           
    def driver_command_dot(self, m, n):
        if n > 1:
            return m.driver_command[n] == m.driver_command[n-1] +  ( (self.Track.delta_s/self.calc_centre_dist_dot(m, n-1)) * (m.delta_driver_command[n-1]) )
        else:
            #use this to leave out members of a set that the constraint doesn't apply to
            return Constraint.Skip
    
    # Define the functions that govern the additional variables
    def calc_rear_slip_angle(self, m, n):
        temp = ((m.lat_vel[n] - (m.L_rear * m.yaw_rate[n])) / m.long_vel[n])
        alpha_r = -atan(temp)
        return alpha_r 
    def calc_rear_LateralForce(self, m, n):
        alpha_r = self.calc_rear_slip_angle(m, n)
        temp = m.C_rear * atan((m.B_rear * alpha_r))
        rear_lateral_Force = m.rear_NormalForce * m.D_rear * sin(temp)
        return rear_lateral_Force
    def calc_front_slip_angle(self, m, n):
        temp = ((m.lat_vel[n] + (m.L_forward * m.yaw_rate[n])) / m.long_vel[n])
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
             return ( m.normal_dist[n] + ((m.L_c/2) * sin(abs(m.local_heading[n]))) + ((m.W_c/2) * cos(m.local_heading[n])) )<= self.Track.width[n] 
        else:
            return Constraint.Skip
    def setRight_width(self, m, n):
        if n < self.Track.N:
            return ( -m.normal_dist[n] + ((m.L_c/2) * sin(abs(m.local_heading[n]))) + ((m.W_c/2) * cos(m.local_heading[n])) )<= self.Track.width[n] 
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
        return c

    # Additional Functions for processing the resultd
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
    
    def optimize(self, initCondition, Parameters=None, finalCondition=None, Temp_Track=None):
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

        # Define the path constraints
        m.setT_Left = Constraint(m.N, rule=self.setLeft_width)
        m.setT_Right = Constraint(m.N, rule=self.setRight_width)

        # Define Friction Constraints
        m.setR_Fric = Constraint(m.N, rule=self.setRear_Fric)
        m.setF_Fric = Constraint(m.N, rule=self.setFront_Fric)

        # Define constraint that forces prgression along the track
        m.progressForward = Constraint(m.N, rule=self.progressForward)

        # Define the Objective Function
        m.Cost = Objective(rule=self.CostFunction)

        # Initialization
        for n in range(1,(self.Track.N)+1):
            # m.delta_steer[n].value = (np.random.random()*(np.pi/3)) - np.pi/6
            # m.delta_driver_command[n].value = (np.random.random()*(2)) - 1
            # m.normal_dist[n].value = (np.random.random()*(np.amax(self.Track.width)*2)) - np.amax(self.Track.width)
            # m.local_heading[n].value = (np.random.random()*(np.pi)) - (np.pi/2)
            m.long_vel[n].value = 5#np.random()
            # m.lat_vel[n].value = (np.random.random()*(2)) - 1
            # m.yaw_rate[n].value = (np.random.random()*np.pi/3) - np.pi/6
            # m.steer[n].value = (np.random.random()*(np.pi/3)) - np.pi/6
            # m.driver_command[n].value = (np.random.random()*(2)) - 1


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
        opt = SolverFactory('ipopt',executable = r"C:\Users\21738394\Oran\Ipopt_3.14.5\bin\ipopt.exe") 
        opt.options["print_level"] = 5 # prints a log with each iteration (you want to this - it's the only way to see progress.)
        opt.options["max_iter"] = 3000000 # maximum number of iterations
        opt.options["max_cpu_time"] = (3600*2) # maximum cpu time in seconds
        opt.options["Tol"] = 1e-6 # the tolerance for feasibility. Considers constraints satisfied when they're within this margin.   

        try:
            results = opt.solve(m, tee = True, symbolic_solver_labels=True)
                  
        except Exception:
            if self.Track.Name is not None:
                if type(self.Track.Name) == str:
                    path = "./" + self.Track.Name
                else:
                    path = "./Optimization Attempt with No Name Track"
            else:
                path = "./Optimization Attempt with No Name Track"
            temp = np.zeros(1)
            csv_path = path + "/FAILED.csv"
            df = pd.DataFrame({"Attempt" : temp})
            df.to_csv(csv_path, index=False)
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
            res = Results(delta_st, delta_T, n, local_heading, x_vel, y_vel, yaw_rate, st, T, cost)
            self.results = res
            if self.Track.Name is not None:
                if type(self.Track.Name) == str:
                    path = "./" + str(self.Track.Name)
                else:
                    path = "./Optimization Attempt with No Name Track"
            else:

                path = "./Optimization Attempt with No Name Track"
            os.makedirs(path , exist_ok=True)
            
            csv_path = path + "/results.csv"
            df = pd.DataFrame({"delta_st" : delta_st, "delta_t" : delta_T, "normal_dist" : n, "local_heading" : local_heading, "x_vel" : x_vel, "y_vel" : y_vel, "yaw_rate" : yaw_rate, "steering_angle" : st, "driver_command" : T})
            df.to_csv(csv_path, index=False)

            plt.plot(s[0:(self.Track.N-1)], s_dot)
            plt.xlabel('S')
            plt.ylabel('S_Dot')
            fig_path = path + "/s_dot.png"
            plt.savefig(fig_path)
            plt.close()
      
            plt.plot(s)
            plt.xlabel('Index')
            plt.ylabel('S')
            fig_path = path + "/s.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(s, delta_st)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Change in Steering angle: Driver Input')
            fig_path = path + "/delta_st.png"
            plt.savefig(fig_path)
            plt.close()
       
            plt.plot(s, delta_T)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Change in Steering angle: Driver Input')
            fig_path = path + "/delta_T.png"
            plt.savefig(fig_path)
            plt.close()

            plt.close()
            plt.plot(s, n)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Orthognal distance from the centre line')
            fig_path = path + "/n.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(s, local_heading)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Local heading')
            fig_path = path + "/mu.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(s, x_vel)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Longitutinal Velocity')
            fig_path = path + "/vx.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(s, y_vel)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Latitudinal Velocity')
            fig_path = path + "/vy.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(s, yaw_rate)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Yaw Rate')
            fig_path = path + "/r.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(s, st)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Steering angle')
            fig_path = path + "/steer.png"
            plt.savefig(fig_path)
            plt.close()

            plt.plot(s, T)
            plt.xlabel('Distance along the centre line')
            plt.ylabel('Driver Command')
            fig_path = path + "/T.png"
            plt.savefig(fig_path)
            plt.close()
          
            text_path = path + "/Results.txt"
            f = open(text_path, 'w')
            if self.Track.Name is not None:
                if type(self.Track.Name) == str:
                    f.write("Optimizing Track with Track Name: " + self.Track.Name + "\n") 
                else: 
                    f.write("Optimizing Track with Track Name: No Name\n") 
            else: 
                f.write("Optimizing Track with Track Name: No Name\n")
            f.write("Final Cost Function: " + str(cost) + "\n")
            f.write(results.solver.status + "\n") # tells you if the solver had any errors/ warnings
            f.write(results.solver.termination_condition + "\n")
            f.close()
            self.Track.exportTrackParameters(path)
            param.exportModelParameters(path)

            self.Track = copy.deepcopy(hold_track)
            return res


if __name__ == "__main__":
    delta_st = 0.0
    delta_T = 0.0
    n = 0.0
    mu = 0.0
    vx = 5.0
    vy = 0.0
    r = 0.0
    st = 0.0
    T = 1.0
    N =2500
    curve = np.zeros(N)
    for i in range(N):
       curve[i] = 0.04 * np.sin(i * 2*np.pi * (1/N))
    # plt.close()
    # plt.plot(curve)
    # plt.show()
    initCondition = State_parameters(delta_st, delta_T, n, mu, vx, vy, r, st, T)
    param = Model_Parameters(Slip_AngleReg=5, Steer_Reg=5)
    width = 10.0
    track = Track(N, 0.1, curve, Width=width, Name=("Test Track Width_" + str(width)))
    opt = Optimizer()
    results = opt.optimize(initCondition, Parameters=param, Temp_Track=track)
 
        
