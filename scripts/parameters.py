import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Tyre_Parameters:
    def __init__(self, B=10.0, C=1.9, D=1.0):
        self.B = B
        self.C = C
        self.D = D

class Kinematic_Bicycle_Model_Parameters:
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
        temp_path = path + "/Kinematic Bicycle Model Parameters.txt"
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

    def model_dynamics_8State(self, state, input, curvature):
        # state = numpy array                  input = numpy array                                            
        # state[0] = Centerline Distance       input[0] = Steering Angle dot
        # state[1] = Orthogonal Distance       input[1] = Driver Command dot
        # state[2] = Local Heading
        # state[3] = Longitudinal Velocity
        # state[4] = Lateral Velocity
        # state[5] = Yaw Rate
        # state[6] = Steering Angle
        # state[7] = Driver Command

        FN_r = ((self.L_rear * self.mass * self.g) / (self.L_forward + self.L_rear))
        FN_f = ((self.L_forward * self.mass * self.g) / (self.L_forward + self.L_rear))

        Fx = (self.C_m*state[7]) - self.C_r0 - (self.C_r2*(state[3]**2))
        alpha_f = -np.arctan((state[4] + self.L_forward*state[5])/state[3]) + state[6]
        Fyf = FN_f*self.frontTyre.D*np.sin(self.frontTyre.C*np.arctan(self.frontTyre.B*alpha_f))
        alpha_r = -np.arctan((state[4] - self.L_rear*state[5])/state[3]) 
        Fyr = FN_r*self.rearTyre.D*np.sin(self.rearTyre.C*np.arctan(self.rearTyre.B*alpha_r))
        Mtv = self.P_tv*(((np.tan(state[6])*state[3])/(self.L_rear + self.L_forward)) - state[5])

        sdot = ((state[3]*np.cos(state[2])) - (state[4]*np.sin(state[2])))/(1 - state[1]*curvature)
        ndot = (state[3]*np.sin(state[2])) + (state[4]*np.cos(state[2]))
        mudot = state[5] - (curvature*sdot)
        vxdot = (1/self.mass)*(Fx - Fyf*np.sin(state[6]) + self.mass*state[4]*state[5])
        vydot = (1/self.mass)*(Fyr + Fyr*np.cos(state[6]) - self.mass*state[3]*state[5])
        rdot = (1/self.I_z) * (Fyf*self.L_forward*np.cos(state[6]) - Fyr*self.L_rear + Mtv)
        steer_dot = input[0]
        T_dot = input[1]
        f = np.array([sdot, ndot, mudot, vxdot, vydot, rdot, steer_dot, T_dot])
        return f

    def model_dynamics_6State(self, state, input, curvature):
        # state = numpy array                  input = numpy array                                            
        # state[0] = Centerline Distance       input[0] = Steering Angle
        # state[1] = Orthogonal Distance       input[1] = Driver Command
        # state[2] = Local Heading
        # state[3] = Longitudinal Velocity
        # state[4] = Lateral Velocity
        # state[5] = Yaw Rate

        FN_r = ((self.L_rear * self.mass * self.g) / (self.L_forward + self.L_rear))
        FN_f = ((self.L_forward * self.mass * self.g) / (self.L_forward + self.L_rear))

        Fx = (self.C_m*input[1]) - self.C_r0 - (self.C_r2*(state[3]**2))
        alpha_f = -np.arctan((state[4] + self.L_forward*state[5])/state[3]) + input[0]
        Fyf = FN_f*self.frontTyre.D*np.sin(self.frontTyre.C*np.arctan(self.frontTyre.B*alpha_f))
        alpha_r = -np.arctan((state[4] - self.L_rear*state[5])/state[3]) 
        Fyr = FN_r*self.rearTyre.D*np.sin(self.rearTyre.C*np.arctan(self.rearTyre.B*alpha_r))
        Mtv = self.P_tv*(((np.tan(input[0])*state[3])/(self.L_rear + self.L_forward)) - state[5])

        sdot = ((state[3]*np.cos(state[2])) - (state[4]*np.sin(state[2])))/(1 - state[1]*curvature)
        ndot = (state[3]*np.sin(state[2])) + (state[4]*np.cos(state[2]))
        mudot = state[5] - (curvature*sdot)
        vxdot = (1/self.mass)*(Fx - Fyf*np.sin(input[0]) + self.mass*state[4]*state[5])
        vydot = (1/self.mass)*(Fyr + Fyf*np.cos(input[0]) - self.mass*state[3]*state[5])
        rdot = (1/self.I_z) * (Fyf*self.L_forward*np.cos(input[0]) - Fyr*self.L_rear + Mtv)

        f = np.array([sdot, ndot, mudot, vxdot, vydot, rdot])
        return f

    def s_dot_6State(self, s, t, n, mu, vx, vy, curvature=0):
        s_dot = ((vx*np.cos(mu)) - (vy*np.sin(mu)))/(1 - n*curvature)
        return s_dot
    
    def n_dot_6State(self,n, t, mu, vx, vy):
        n_dot = (vx*np.sin(mu)) + (vy*np.cos(mu))
        return n_dot
    
    def mu_dot_6State(self, mu, t, n, vx, vy, r, curvature=0):
        mu_dot = r - (curvature*((vx*np.cos(mu)) - (vy*np.sin(mu)))/(1 - n*curvature))
        return mu_dot
    
    def vx_dot_6State(self, vx, t, vy, r, steer, T,):
        FN_f = ((self.L_forward * self.mass * self.g) / (self.L_forward + self.L_rear))
        Fx = (self.C_m*T) - self.C_r0 - (self.C_r2*(vx**2))
        alpha_f = -np.arctan((vy + self.L_forward*r)/vx) + steer
        Fyf = FN_f*self.frontTyre.D*np.sin(self.frontTyre.C*np.arctan(self.frontTyre.B*alpha_f))

        vx_dot = (1/self.mass)*(Fx - Fyf*np.sin(steer) + self.mass*vy*r)
        return vx_dot
    
    def vy_dot_6State(self, vy, t, vx, r, steer,):
        FN_r = ((self.L_rear * self.mass * self.g) / (self.L_forward + self.L_rear))
        FN_f = ((self.L_forward * self.mass * self.g) / (self.L_forward + self.L_rear))

        alpha_f = -np.arctan((vy + self.L_forward*r)/vx) + steer
        Fyf = FN_f*self.frontTyre.D*np.sin(self.frontTyre.C*np.arctan(self.frontTyre.B*alpha_f))
        alpha_r = -np.arctan((vy - self.L_rear*r)/vx) 
        Fyr = FN_r*self.rearTyre.D*np.sin(self.rearTyre.C*np.arctan(self.rearTyre.B*alpha_r))

        vy_dot = (1/self.mass)*(Fyr + Fyf*np.cos(steer) - self.mass*vx*r)
        return vy_dot
    
    def r_dot_6State(self, r, t, vx, vy, steer):
        FN_r = ((self.L_rear * self.mass * self.g) / (self.L_forward + self.L_rear))
        FN_f = ((self.L_forward * self.mass * self.g) / (self.L_forward + self.L_rear))

        alpha_f = -np.arctan((vy + self.L_forward*r)/vx) + steer
        Fyf = FN_f*self.frontTyre.D*np.sin(self.frontTyre.C*np.arctan(self.frontTyre.B*alpha_f))
        alpha_r = -np.arctan((vy - self.L_rear*r)/vx) 
        Fyr = FN_r*self.rearTyre.D*np.sin(self.rearTyre.C*np.arctan(self.rearTyre.B*alpha_r))
        Mtv = self.P_tv*(((np.tan(steer)*vx)/(self.L_rear + self.L_forward)) - r)

        r_dot = (1/self.I_z) * (Fyf*self.L_forward*np.cos(steer) - Fyr*self.L_rear + Mtv)
        return r_dot

    def step_response_6State(self, init_conditions=[0,0,0,0,0,0], magnitude=1, maxtime=10, timestep=0.01, curvature=0):
        time = np.linspace(0, maxtime, (int(maxtime/timestep)))
        
        steer = np.zeros((len(time)))
        T_step = np.zeros((len(time)))
        T_step[(int(1/timestep)+1):] = 1*magnitude

        s0 = init_conditions[0]
        n0 = init_conditions[1]
        mu0 = init_conditions[2]
        vx0 = init_conditions[3]
        vy0 = init_conditions[4]
        r0 = init_conditions[5]

        s = np.zeros(len(time))
        n = np.zeros(len(time))
        mu = np.zeros(len(time))
        vx = np.zeros(len(time))
        vy = np.zeros(len(time))
        r = np.zeros(len(time))

        for i in range((len(time))-1):
            
            centerline_dist = odeint(self.s_dot_6State, s0, [0, timestep], args=(n[i], mu[i], vx[i], vy[i], curvature))
            orthogonal_dist = odeint(self.n_dot_6State, n0, [0, timestep], args=(mu[i], vx[i], vy[i]))
            heading = odeint(self.mu_dot_6State, mu0, [0, timestep], args=(n[i], vx[i], vy[i], r[i], curvature))
            long_velocity = odeint(self.vx_dot_6State, vx0, [0, timestep], args=(vy[i], r[i], steer[i], T_step[i]))
            lat_velocity = odeint(self.vy_dot_6State, vy0, [0, timestep], args=(vx[i], r[i], steer[i]))
            yaw_rate = odeint(self.r_dot_6State, r0, [0, timestep], args= (vx[i], vy[i], steer[i]))

            s0 = centerline_dist[-1]
            n0 = orthogonal_dist[-1]
            mu0 = heading[-1]
            vx0 = long_velocity[-1]
            vy0 = lat_velocity[-1]
            r0 = yaw_rate[-1]

            s[i+1] = s0
            n[i+1] = n0
            mu[i+1] = mu0
            vx[i+1] = vx0
            vy[i+1] = vy0
            r[i+1] = r0

        plt.subplots(2,1,sharex=True)
        plt.plot(time,vx,'b-',linewidth=3)
        plt.ylabel('Longitudinal Velocity (m/s)')
        plt.legend(['Velocity'])
        plt.subplots(2,1,sharex=True)
        plt.plot(time,T_step,'r--',linewidth=3)
        plt.ylabel('Driver Command')    
        plt.legend(['Driver Command'])
        plt.xlabel('Time (sec)')
        plt.show()
            
class Single_Track_Model_Parameters:
    def __init__(self, mu=1.0489, C_Sf=4.718, C_Sr=5.4562, lf=0.15875, lr=0.17145, h=0.074, m=3.74, I=0.04712, s_min=0.4189, s_max=0.4189, sv_min=-3.2, sv_max=3.2, v_switch=7.319, a_max=9.51, v_min=-5.0, v_max=20.0, width=0.31, length=0.58):
        self.mu = mu # surface friction coefficient
        self.C_Sf = C_Sf # Cornering stiffness coefficient, front
        self.C_Sr = C_Sr # Cornering stiffness coefficient, rear
        self.lf = lf # Distance from center of gravity to front axle
        self.lr = lr # Distance from center of gravity to rear axle
        self.h = h # Height of center of gravity
        self.m = m # Total mass of the vehicle
        self.I = I # Moment of inertial of the entire vehicle about the z axis
        self.s_min = s_min # Minimum steering angle constraint
        self.s_max = s_max # Maximum steering angle constraint
        self.sv_min = sv_min # Minimum steering velocity constraint
        self.sv_max = sv_max # Maximum steering velocity constraint
        self.v_switch = v_switch # Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
        self.a_max = a_max # Maximum longitudinal acceleration
        self.v_min = v_min # Minimum longitudinal velocity
        self.v_max = v_max # Maximum longitudinal velocity
        self.width = width # width of the vehicle in meters
        self.length = length # length of the vehicle in meters

    def exportModelParameters(self, path):
        line1 = "Car Length: " + str(self.length) + "\n"
        line2 = "Car Width: " + str(self.width) + "\n"
        line3 = "Surface friction coefficient: " + str(self.mu) + "\n"
        line4 = "Cornering stiffness coefficient, front: " + str(self.C_Sf) + "\n"
        line5 = "Cornering stiffness coefficient, rear: " + str(self.C_Sr) + "\n"
        line6 = "Distance from center of gravity to front axle: " + str(self.lf) + "\n"
        line7 = "Distance from center of gravity to rear axle: " + str(self.lr) + "\n"
        line8 = "Height of center of gravity: " + str(self.h) + "\n"
        line9 = "Total mass of the vehicle: " + str(self.m) + "\n"
        line10 = "Moment of inertial of the entire vehicle about the z axis: " + str(self.I) + "\n"
        line11 = "Minimum steering angle constraint: " + str(self.s_min) + "\n"
        line12 = "Maximum steering angle constraint: " + str(self.s_max) + "\n"
        line13 = "Minimum steering velocity constraint: " + str(self.sv_min) + "\n"
        line14 = "Maximum steering velocity constraint: " + str(self.sv_max) + "\n"
        line15 = "Switching velocity (velocity at which the acceleration is no longer able to create wheel spin): " + str( self.v_switch) + "\n"
        line16 = "Maximum longitudinal acceleration: " + str(self.a_max) + "\n"
        line17 = "Minimum longitudinal velocity: " + str(self.v_min) + "\n"
        line18 = "Maximum longitudinal velocity: " + str(self.v_max) + "\n"

        temp_path = path + "/Single Track Model Parameters.txt"
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
        f.close()

    def accl_constraints(self, vel, accl, v_switch, a_max, v_min, v_max):
        """
        Acceleration constraints, adjusts the acceleration based on constraints
            Args:
                vel (float): current velocity of the vehicle
                accl (float): unconstraint desired acceleration
                v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity
            Returns:
                accl (float): adjusted acceleration
        """

        # positive accl limit
        if vel > v_switch:
            pos_limit = a_max*v_switch/vel
        else:
            pos_limit = a_max

        # accl limit reached?
        if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
            accl = 0.
        elif accl <= -a_max:
            accl = -a_max
        elif accl >= pos_limit:
            accl = pos_limit

        return accl

    def steering_constraint(self, steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
        """
        Steering constraints, adjusts the steering velocity based on constraints
            Args:
                steering_angle (float): current steering_angle of the vehicle
                steering_velocity (float): unconstraint desired steering_velocity
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity
            Returns:
                steering_velocity (float): adjusted steering velocity
        """

        # constraint steering velocity
        if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
            steering_velocity = 0.
        elif steering_velocity <= sv_min:
            steering_velocity = sv_min
        elif steering_velocity >= sv_max:
            steering_velocity = sv_max

        return steering_velocity

    def model_dynamics(self, state, input):
        # """
        # Single Track Dynamic Vehicle Dynamics.
        #     Args:
        #         x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
        #             x1: x position in global coordinates
        #             x2: y position in global coordinates
        #             x3: steering angle of front wheels
        #             x4: velocity in x direction
        #             x5: yaw angle
        #             x6: yaw rate
        #             x7: slip angle at vehicle center
        #         u (numpy.ndarray (2, )): control input vector (u1, u2)
        #             u1: steering angle velocity of front wheels
        #             u2: longitudinal acceleration
        #     Returns:
        #         f (numpy.ndarray): right hand side of differential equations
        # """

        # gravity constant m/s^2
        g = 9.81

        # constraints
        u = np.array([self.steering_constraint(state[2], input[0], self.s_min, self.s_max, self.sv_min, self.sv_max), self.accl_constraints(state[3], input[1], self.v_switch, self.a_max, self.v_min, self.v_max)])

        # system dynamics
        f = np.array([state[3]*np.cos(state[6] + state[4]),
            state[3]*np.sin(state[6] + state[4]),
            u[0],
            u[1],
            state[5],
            -self.mu*self.m/(state[3]*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(g*self.lr-u[1]*self.h) + self.lr**2*self.C_Sr*(g*self.lf + u[1]*self.h))*state[5] \
                +self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(g*self.lf + u[1]*self.h) - self.lf*self.C_Sf*(g*self.lr - u[1]*self.h))*state[6] \
                +self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(g*self.lr - u[1]*self.h)*state[2],
            (self.mu/(state[3]**2*(self.lr+self.lf))*(self.C_Sr*(g*self.lf + u[1]*self.h)*self.lr - self.C_Sf*(g*self.lr - u[1]*self.h)*self.lf)-1)*state[5] \
                -self.mu/(state[3]*(self.lr+self.lf))*(self.C_Sr*(g*self.lf + u[1]*self.h) + self.C_Sf*(g*self.lr-u[1]*self.h))*state[6] \
                +self.mu/(state[3]*(self.lr+self.lf))*(self.C_Sf*(g*self.lr-u[1]*self.h))*state[2]])

        return f


        step_response_6State()
if __name__ == "__main__":
    car = Kinematic_Bicycle_Model_Parameters()
    #car.step_response_6State()
