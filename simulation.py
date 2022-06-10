import numpy as np
import scipy.integrate as integrate
from optimization import Model_Parameters
from autocar import auto_car

def trapeziodal_integration(y, constant_step, initial_conditions=0):
    temp = initial_conditions
    for i in range(len(y)-1):
        temp += (y[i] + y[i+1]) * (constant_step/2)
    return temp

class environment:
    def __init__(self, initialConditions=None, modelParameters=None):
        self.timestep = 0.1
        self.maxtime = 100
        self.update_time = 0.5
        self.sample_time = 0.05
        self.state_dot_list = []
        if modelParameters is None:
            self.modelParam = Model_Parameters()
        else:
            self.modelParam = modelParameters
        if initialConditions is None:
            self.initialConditions = np.array([0, 0, 0, 0.05, 0, 0]) # my Model; 
        else:
            self.initialConditions = initialConditions

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

    def f110_car_model_dynamics(self, state, input, modelParam):

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
        u = np.array([self.steering_constraint(state[2], input[0], modelParam.s_min, modelParam.s_max, modelParam.sv_min, modelParam.sv_max), self.accl_constraints(state[3], input[1], modelParam.v_switch, modelParam.a_max, modelParam.v_min, modelParam.v_max)])

        # system dynamics
        f = np.array([state[3]*np.cos(state[6] + state[4]),
            state[3]*np.sin(state[6] + state[4]),
            u[0],
            u[1],
            state[5],
            -modelParam.mu*modelParam.m/(state[3]*modelParam.I*(modelParam.lr+modelParam.lf))*(modelParam.lf**2*modelParam.C_Sf*(g*modelParam.lr-u[1]*modelParam.h) + modelParam.lr**2*modelParam.C_Sr*(g*modelParam.lf + u[1]*modelParam.h))*state[5] \
                +modelParam.mu*modelParam.m/(modelParam.I*(modelParam.lr+modelParam.lf))*(modelParam.lr*modelParam.C_Sr*(g*modelParam.lf + u[1]*modelParam.h) - modelParam.lf*modelParam.C_Sf*(g*modelParam.lr - u[1]*modelParam.h))*state[6] \
                +modelParam.mu*modelParam.m/(modelParam.I*(modelParam.lr+modelParam.lf))*modelParam.lf*modelParam.C_Sf*(g*modelParam.lr - u[1]*modelParam.h)*state[2],
            (modelParam.mu/(state[3]**2*(modelParam.lr+modelParam.lf))*(modelParam.C_Sr*(g*modelParam.lf + u[1]*modelParam.h)*modelParam.lr - modelParam.C_Sf*(g*modelParam.lr - u[1]*modelParam.h)*modelParam.lf)-1)*state[5] \
                -modelParam.mu/(state[3]*(modelParam.lr+modelParam.lf))*(modelParam.C_Sr*(g*modelParam.lf + u[1]*modelParam.h) + modelParam.C_Sf*(g*modelParam.lr-u[1]*modelParam.h))*state[6] \
                +modelParam.mu/(state[3]*(modelParam.lr+modelParam.lf))*(modelParam.C_Sf*(g*modelParam.lr-u[1]*modelParam.h))*state[2]])

        return f

    def car_model_dynamics(self, state, input, modelParam, curvature):
        # state = numpy array                  input = numpy array                                            
        # state[0] = Centerline Distance       input[0] = Steering Angle dot
        # state[1] = Orthogonal Distance       input[1] = Driver Command dot
        # state[2] = Local Heading
        # state[3] = Longitudinal Velocity
        # state[4] = Lateral Velocity
        # state[5] = Yaw Rate
        # state[6] = Steering Angle
        # state[7] = Driver Command

        FN_r = ((modelParam.L_rear * modelParam.mass * modelParam.g) / (modelParam.L_forward + modelParam.L_rear))
        FN_f = ((modelParam.L_forward * modelParam.mass * modelParam.g) / (modelParam.L_forward + modelParam.L_rear))

        Fx = (modelParam.C_m*state[7]) - modelParam.C_r0 - (modelParam.C_r2*(state[3]**2))
        alpha_f = -np.arctan((state[4] + modelParam.L_forward*state[5])/state[3]) + state[6]
        Fyf = FN_f*modelParam.frontTyre.D*np.sin(modelParam.frontTyre.C*np.arctan(modelParam.frontTyre.B*alpha_f))
        alpha_r = -np.arctan((state[4] - modelParam.L_rear*state[5])/state[3]) 
        Fyr = FN_r*modelParam.rearTyre.D*np.sin(modelParam.rearTyre.C*np.arctan(modelParam.rearTyre.B*alpha_r))
        Mtv = modelParam.P_tv*(((np.tan(state[6])*state[3])/(modelParam.L_rear + modelParam.L_forward)) - state[5])

        sdot = ((state[3]*np.cos(state[2])) - (state[4]*np.sin(state[2])))/(1 - state[1]*curvature)
        ndot = (state[3]*np.sin(state[2])) + (state[4]*np.cos(state[2]))
        mudot = state[5] - (curvature*sdot)
        vxdot = (1/modelParam.mass)*(Fx - Fyf*np.sin(state[6]) + modelParam.mass*state[4]*state[5])
        vydot = (1/modelParam.mass)*(Fyr + Fyr*np.cos(state[6]) - modelParam.mass*state[3]*state[5])
        rdot = (1/modelParam.I_z) * (Fyf*modelParam.L_forward*np.cos(state[6]) - Fyr*modelParam.L_rear + Mtv)
        steer_dot = input[0]
        T_dot = input[1]
        f = np.array([sdot, ndot, mudot, vxdot, vydot, rdot, steer_dot, T_dot])
        return f

    def car_model_dynamics_6state(self, state, input, modelParam, curvature):
        # state = numpy array                  input = numpy array                                            
        # state[0] = Centerline Distance       input[0] = Steering Angle
        # state[1] = Orthogonal Distance       input[1] = Driver Command
        # state[2] = Local Heading
        # state[3] = Longitudinal Velocity
        # state[4] = Lateral Velocity
        # state[5] = Yaw Rate

        FN_r = ((modelParam.L_rear * modelParam.mass * modelParam.g) / (modelParam.L_forward + modelParam.L_rear))
        FN_f = ((modelParam.L_forward * modelParam.mass * modelParam.g) / (modelParam.L_forward + modelParam.L_rear))

        Fx = (modelParam.C_m*input[1]) - modelParam.C_r0 - (modelParam.C_r2*(state[3]**2))
        alpha_f = -np.arctan((state[4] + modelParam.L_forward*state[5])/state[3]) + input[0]
        Fyf = FN_f*modelParam.frontTyre.D*np.sin(modelParam.frontTyre.C*np.arctan(modelParam.frontTyre.B*alpha_f))
        alpha_r = -np.arctan((state[4] - modelParam.L_rear*state[5])/state[3]) 
        Fyr = FN_r*modelParam.rearTyre.D*np.sin(modelParam.rearTyre.C*np.arctan(modelParam.rearTyre.B*alpha_r))
        Mtv = modelParam.P_tv*(((np.tan(input[0])*state[3])/(modelParam.L_rear + modelParam.L_forward)) - state[5])

        sdot = ((state[3]*np.cos(state[2])) - (state[4]*np.sin(state[2])))/(1 - state[1]*curvature)
        ndot = (state[3]*np.sin(state[2])) + (state[4]*np.cos(state[2]))
        mudot = state[5] - (curvature*sdot)
        vxdot = (1/modelParam.mass)*(Fx - Fyf*np.sin(input[0]) + modelParam.mass*state[4]*state[5])
        vydot = (1/modelParam.mass)*(Fyr + Fyr*np.cos(input[0]) - modelParam.mass*state[3]*state[5])
        rdot = (1/modelParam.I_z) * (Fyf*modelParam.L_forward*np.cos(input[0]) - Fyr*modelParam.L_rear + Mtv)

        f = np.array([sdot, ndot, mudot, vxdot, vydot, rdot])
        return f


    def track_curvarture(self, s):
        # for i in range((autocar.N_samples-1)):
        #     if ( (autocar.track.delta_s * i ) <= s) and ( (autocar.track.delta_s * (i+1) ) >= s):
        #         curvature = self.linearly_interpolate((autocar.track.delta_s * i ), (autocar.track.delta_s * (i+1) ), s, autocar.track.delta_s)
        #         return curvature
            
        # raise Exception("Unable to find track curvature at current position")
        return 0

    def coordinate_transform(self, state):
        #pass
        return state

   
    def update_car_dynamics(self, current_state, input):
        current_curvi_states = self.coordinate_transform(current_state)  
        curvature = self.track_curvarture(current_curvi_states[0]) 
        #state_dot = self.car_model_dynamics(current_curvi_states, input, self.modelParam, curvature)
        state_dot = self.car_model_dynamics_6state(current_curvi_states, input, self.modelParam, curvature)
        self.state_dot_list.append(state_dot)
        
        new_state = np.zeros(len(current_state))
        temp = np.array(self.state_dot_list)
        #print(temp.shape)
        for i in range(len(new_state)):
            new_state[i] = trapeziodal_integration(temp[:, i], self.timestep, initial_conditions=self.initialConditions[i])
            #new_state[i] =  integrate.trapezoid(temp[:,i], dx=self.timestep)
            #new_state[i] =  integrate.romb(temp[:,i], dx=self.timestep)
            #new_state[i] =  integrate.simpson(temp[:,i], dx=self.timestep)
            #new_state[i] = current_curvi_states[i] + self.timestep*state_dot[i]
        return new_state

    def display(self, state, input, time):
        print("Longitudinal Velocity at time {}, is {}\n".format(time, state[3]))
        #print("Driver Command at time {}, is {}\n".format(time, input[1]))




    def main(self, autocar):
        # Environment loop
        current_state = self.initialConditions
        current_input = np.array([0, 1])

        for step in range(int(self.maxtime/self.timestep)):
            
            current_time = step * self.timestep
            if current_time == self.maxtime:
                break
            
            if((current_time % self.update_time) == 0):
                autocar.update_local_plan(current_state) # At every update time, rerun the local planner, and recreate the local plan
            if((current_time % self.sample_time) == 0):
                current_input = autocar.car_control(current_state, current_time) # At every sampling time, give a input to controls of the car

            current_state = self.update_car_dynamics(current_state, current_input) #Updates and keeps track of the sim Car dynamics
            self.display(current_state, current_input, current_time)

if __name__ == "__main__":

    sim_env = environment()
    agent = auto_car(None)
    sim_env.main(agent)



