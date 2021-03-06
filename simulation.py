from turtle import distance
import numpy as np
import scipy.integrate as integrate
#import optimization as opt'
from mapping import map
import LibFunctions as lib
from optimization import Track
from optimization import State_parameters
from optimization import Model_Parameters
from autocar import auto_car

class environment:
    def __init__(self, Track, initialConditions=None, modelParameters=None):
        self.track = Track
        self.timestep = 0.0001
        self.maxtime = 100
        self.update_time = 0.5
        self.sample_time = 0.05
        self.state_dot_list = []
        self.state_list = []
        if modelParameters is None:
            self.modelParam = Model_Parameters(Slip_AngleReg=5, Steer_Reg=5, Max_Tyre_Force=4.0)
        else:
            self.modelParam = modelParameters
        if initialConditions is None:
            self.initialConditions = np.array([0, 0, 0, 5, 0, 0]) # my Model; 
        else:
            self.initialConditions = initialConditions
        self.state_list.append(self.initialConditions)
    #TODO: Write a model parameter class for the f110 car model
    
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

    def f110_car_model_dynamics(self, state, input):

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
        u = np.array([self.steering_constraint(state[2], input[0], self.modelParam.s_min, self.modelParam.s_max, self.modelParam.sv_min, self.modelParam.sv_max), self.accl_constraints(state[3], input[1], self.modelParam.v_switch, self.modelParam.a_max, self.modelParam.v_min, self.modelParam.v_max)])

        # system dynamics
        f = np.array([state[3]*np.cos(state[6] + state[4]),
            state[3]*np.sin(state[6] + state[4]),
            u[0],
            u[1],
            state[5],
            -self.modelParam.mu*self.modelParam.m/(state[3]*self.modelParam.I*(self.modelParam.lr+self.modelParam.lf))*(self.modelParam.lf**2*self.modelParam.C_Sf*(g*self.modelParam.lr-u[1]*self.modelParam.h) + self.modelParam.lr**2*self.modelParam.C_Sr*(g*self.modelParam.lf + u[1]*self.modelParam.h))*state[5] \
                +self.modelParam.mu*self.modelParam.m/(self.modelParam.I*(self.modelParam.lr+self.modelParam.lf))*(self.modelParam.lr*self.modelParam.C_Sr*(g*self.modelParam.lf + u[1]*self.modelParam.h) - self.modelParam.lf*self.modelParam.C_Sf*(g*self.modelParam.lr - u[1]*self.modelParam.h))*state[6] \
                +self.modelParam.mu*self.modelParam.m/(self.modelParam.I*(self.modelParam.lr+self.modelParam.lf))*self.modelParam.lf*self.modelParam.C_Sf*(g*self.modelParam.lr - u[1]*self.modelParam.h)*state[2],
            (self.modelParam.mu/(state[3]**2*(self.modelParam.lr+self.modelParam.lf))*(self.modelParam.C_Sr*(g*self.modelParam.lf + u[1]*self.modelParam.h)*self.modelParam.lr - self.modelParam.C_Sf*(g*self.modelParam.lr - u[1]*self.modelParam.h)*self.modelParam.lf)-1)*state[5] \
                -self.modelParam.mu/(state[3]*(self.modelParam.lr+self.modelParam.lf))*(self.modelParam.C_Sr*(g*self.modelParam.lf + u[1]*self.modelParam.h) + self.modelParam.C_Sf*(g*self.modelParam.lr-u[1]*self.modelParam.h))*state[6] \
                +self.modelParam.mu/(state[3]*(self.modelParam.lr+self.modelParam.lf))*(self.modelParam.C_Sf*(g*self.modelParam.lr-u[1]*self.modelParam.h))*state[2]])

        return f

    def car_model_dynamics(self, state, input, curvature):
        # state = numpy array                  input = numpy array                                            
        # state[0] = Centerline Distance       input[0] = Steering Angle dot
        # state[1] = Orthogonal Distance       input[1] = Driver Command dot
        # state[2] = Local Heading
        # state[3] = Longitudinal Velocity
        # state[4] = Lateral Velocity
        # state[5] = Yaw Rate
        # state[6] = Steering Angle
        # state[7] = Driver Command

        FN_r = ((self.modelParam.L_rear * self.modelParam.mass * self.modelParam.g) / (self.modelParam.L_forward + self.modelParam.L_rear))
        FN_f = ((self.modelParam.L_forward * self.modelParam.mass * self.modelParam.g) / (self.modelParam.L_forward + self.modelParam.L_rear))

        Fx = (self.modelParam.C_m*state[7]) - self.modelParam.C_r0 - (self.modelParam.C_r2*(state[3]**2))
        alpha_f = -np.arctan((state[4] + self.modelParam.L_forward*state[5])/state[3]) + state[6]
        Fyf = FN_f*self.modelParam.frontTyre.D*np.sin(self.modelParam.frontTyre.C*np.arctan(self.modelParam.frontTyre.B*alpha_f))
        alpha_r = -np.arctan((state[4] - self.modelParam.L_rear*state[5])/state[3]) 
        Fyr = FN_r*self.modelParam.rearTyre.D*np.sin(self.modelParam.rearTyre.C*np.arctan(self.modelParam.rearTyre.B*alpha_r))
        Mtv = self.modelParam.P_tv*(((np.tan(state[6])*state[3])/(self.modelParam.L_rear + self.modelParam.L_forward)) - state[5])

        sdot = ((state[3]*np.cos(state[2])) - (state[4]*np.sin(state[2])))/(1 - state[1]*curvature)
        ndot = (state[3]*np.sin(state[2])) + (state[4]*np.cos(state[2]))
        mudot = state[5] - (curvature*sdot)
        vxdot = (1/self.modelParam.mass)*(Fx - Fyf*np.sin(state[6]) + self.modelParam.mass*state[4]*state[5])
        vydot = (1/self.modelParam.mass)*(Fyr + Fyr*np.cos(state[6]) - self.modelParam.mass*state[3]*state[5])
        rdot = (1/self.modelParam.I_z) * (Fyf*self.modelParam.L_forward*np.cos(state[6]) - Fyr*self.modelParam.L_rear + Mtv)
        steer_dot = input[0]
        T_dot = input[1]
        f = np.array([sdot, ndot, mudot, vxdot, vydot, rdot, steer_dot, T_dot])
        return f

    def car_model_dynamics_6state(self, state, input, curvature):
        # state = numpy array                  input = numpy array                                            
        # state[0] = Centerline Distance       input[0] = Steering Angle
        # state[1] = Orthogonal Distance       input[1] = Driver Command
        # state[2] = Local Heading
        # state[3] = Longitudinal Velocity
        # state[4] = Lateral Velocity
        # state[5] = Yaw Rate

        FN_r = ((self.modelParam.L_rear * self.modelParam.mass * self.modelParam.g) / (self.modelParam.L_forward + self.modelParam.L_rear))
        FN_f = ((self.modelParam.L_forward * self.modelParam.mass * self.modelParam.g) / (self.modelParam.L_forward + self.modelParam.L_rear))

        Fx = (self.modelParam.C_m*input[1]) - self.modelParam.C_r0 - (self.modelParam.C_r2*(state[3]**2))
        alpha_f = -np.arctan((state[4] + self.modelParam.L_forward*state[5])/state[3]) + input[0]
        Fyf = FN_f*self.modelParam.frontTyre.D*np.sin(self.modelParam.frontTyre.C*np.arctan(self.modelParam.frontTyre.B*alpha_f))
        alpha_r = -np.arctan((state[4] - self.modelParam.L_rear*state[5])/state[3]) 
        Fyr = FN_r*self.modelParam.rearTyre.D*np.sin(self.modelParam.rearTyre.C*np.arctan(self.modelParam.rearTyre.B*alpha_r))
        Mtv = self.modelParam.P_tv*(((np.tan(input[0])*state[3])/(self.modelParam.L_rear + self.modelParam.L_forward)) - state[5])

        sdot = ((state[3]*np.cos(state[2])) - (state[4]*np.sin(state[2])))/(1 - state[1]*curvature)
        ndot = (state[3]*np.sin(state[2])) + (state[4]*np.cos(state[2]))
        mudot = state[5] - (curvature*sdot)
        vxdot = (1/self.modelParam.mass)*(Fx - Fyf*np.sin(input[0]) + self.modelParam.mass*state[4]*state[5])
        vydot = (1/self.modelParam.mass)*(Fyr + Fyr*np.cos(input[0]) - self.modelParam.mass*state[3]*state[5])
        rdot = (1/self.modelParam.I_z) * (Fyf*self.modelParam.L_forward*np.cos(input[0]) - Fyr*self.modelParam.L_rear + Mtv)

        f = np.array([sdot, ndot, mudot, vxdot, vydot, rdot])
        return f
    
    def linearly_interpolate(self, yA, yB, xA, xB, xC):
        gradient = (yB - yA)/(xB - xA)
        diff = xC - xA
        value = (gradient*diff) + yA
        return value

    def track_curvarture(self, s):
        for i in range((self.track.N)):
            if ( (self.track.delta_s * i ) <= s) and ( (self.track.delta_s * (i+1) ) >= s):
                curvature = self.linearly_interpolate(self.track.curvature[i] , self.track.curvature[i+1], (self.track.delta_s*i), (self.track.delta_s*(i+1)), s)
                return curvature
            
        raise Exception("Unable to find track curvature at current position")

    def find_angle(self, A, B, C):
        # RETURNS THE ANGLE B??C
        vec_AB = A - B
        vec_AC = A - C 
        dot = vec_AB.dot(vec_AC)
        #dot = (A[0] - C[0])*(A[0] - B[0]) + (A[1] - C[1])*(A[1] - B[1])
        magnitude_AB = np.linalg.norm(vec_AB)
        magnitude_AC = np.linalg.norm(vec_AC)

        angle = np.arccos(dot/(magnitude_AB*magnitude_AC))
        return angle

    def transform_XY_to_NS(self, x, y):
        distances = np.zeros(len(self.track.N))
        for i in range(len(self.track.N)):
            distances[i] = lib.get_distance(self.track.track_points[:, i], np.array[x, y])
        
        i_s = np.argmax(distances)
        if i_s >= 1:
            if distances[i_s-1] < distances[i_s+1]:
                i_s2 = i_s - 1
                ang = self.find_angle(self.track.track_points[:, i_s2], self.track.track_points[:, i_s], np.array[x, y])
                actual_s = (i_s2 * self.track.delta_s) + (np.cos(ang) * distances[i_s2])
                actual_n = (np.sin(ang) * distances[i_s2])
            elif  distances[i_s-1] > distances[i_s+1]:
                i_s2 = i_s + 1
                ang = self.find_angle(self.track.track_points[:, i_s], self.track.track_points[:, i_s2], np.array[x, y])
                actual_s = (i_s * self.track.delta_s) + (np.cos(ang) * distances[i_s])
                actual_n = (np.sin(ang) * distances[i_s])
            else:
                raise Exception("Error")
        else:
            i_s2 = i_s + 1
            ang = self.find_angle(self.track.track_points[:, i_s], self.track.track_points[:, i_s2], np.array[x, y])
            actual_s = (i_s * self.track.delta_s) + (np.cos(ang) * distances[i_s])
            actual_n = (np.sin(ang) * distances[i_s])
        return (actual_s, actual_n)   

    def transform_NS_to_XY(self, s, n):
        for i in range((self.track.N-1)):
            if ( (self.track.delta_s * i ) <= s) and ( (self.track.delta_s * (i+1) ) >= s):
                yaw_angle = self.linearly_interpolate(self.track.yaw_angle[i], self.track.yaw_angle[i+1], (self.track.delta_s*i), (self.track.delta_s*(i+1)), s)
                track_x = self.linearly_interpolate(self.track.track_points[0, i], self.track.track_points[0, i+1], (self.track.delta_s*i), (self.track.delta_s*(i+1)), s)
                track_y = self.linearly_interpolate(self.track.track_points[1, i], self.track.track_points[1, i+1], (self.track.delta_s*i), (self.track.delta_s*(i+1)), s)
                break
        actual_X = -n*np.sin(yaw_angle) + track_x
        actual_Y = n*np.cos(yaw_angle) + track_y

        return (actual_X, actual_Y)
    
    def coordinate_transform(self, state):
        # state = numpy array                                                           
        # state[0] = Centerline Distance       
        # state[1] = Orthogonal Distance       
        # state[2] = Local Heading
        # state[3] = Longitudinal Velocity
        # state[4] = Lateral Velocity
        # state[5] = Yaw Rate
        return state
   
    def update_car_dynamics(self, current_state, input):
        current_curvi_states = self.coordinate_transform(current_state)  
        curvature = self.track_curvarture(current_curvi_states[0]) 
        state_dot = self.car_model_dynamics_6state(current_curvi_states, input, curvature)
        self.state_dot_list.append(state_dot)
        
        new_state = np.zeros(len(current_state))

        for i in range(len(new_state)):
            #new_state[i] =  integrate.trapezoid(temp[:,i], dx=self.timestep)
            #new_state[i] =  integrate.romb(temp[:,i], dx=self.timestep)
            #new_state[i] =  integrate.simpson(temp[:,i], dx=self.timestep)
            new_state[i] = current_curvi_states[i] + self.timestep*state_dot[i]
            #new_state[i] = (current_curvi_states[i] + state_dot[i]) * (self.timestep/2)
        self.state_list.append(new_state)
        return new_state

    def main(self, autocar):
        # Environment loop
        current_state = self.initialConditions
        current_input = np.array([0, 1])
        last_update  = 0
        last_sample = 0
        for step in range(int(self.maxtime/self.timestep)):
            
            current_time = step * self.timestep

            if current_time == self.maxtime:
                break
            
            if((current_time - last_update) >= self.update_time):
                last_update = current_time
                autocar.update_local_plan(current_state) # At every update time, rerun the local planner, and recreate the local plan
            if((current_time - last_sample) >= self.update_time):
                last_sample = current_time
                current_input = autocar.car_control(current_state, current_time) # At every sampling time, give a input to controls of the car

            current_state = self.update_car_dynamics(current_state, current_input) #Updates and keeps track of the sim Car dynamics
            
            if current_state[0] >= (self.track.delta_s*self.track.N):
                print("Track Finished in Time {} seconds".format(current_time))
                break
            if current_state[0] < 0:
                print("Error car is going backwards at time {} seconds".format(current_time))
                break

if __name__ == "__main__":
    m = map('columbia_1') #Works
    
  
    init_conditions = State_parameters(0, 0, 0, 0, 0.1, 0, 0, 0, 1)
    track = m.generate_track(0.1)
    sim_env = environment(track, initialConditions=np.array([0, 0, 0, 0.1, 0, 0]) )
    agent = auto_car(track, initialConditions=init_conditions)
    sim_env.main(agent)



