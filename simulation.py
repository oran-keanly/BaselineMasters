import numpy as np

track = Track
timestep = 0.05
maxtime = 100
update_time = 0.5
sample_time = 0.05

def f110_car_model_dynamics(state, input):

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
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = np.array([x[3]*np.cos(x[6] + x[4]),
        x[3]*np.sin(x[6] + x[4]),
        u[0],
        u[1],
        x[5],
        -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
            +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
            +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
        (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
            -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
            +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

    return f

def car_model_dynamics(state, input, modelParam, curvature):
    # state = numpy array                  input = numpy array
    # state[0] = Centerline Distance       input[0] = Steering Angle dot
    # state[1] = Orthogonal Distance       input[1] = Driver Command dot
    # state[2] = Local Heading
    # state[3] = Longitudinal Velocity
    # state[4] = Lateral Velocity
    # state[5] = Yaw Rate
    # state[6] = Steering Angle
    # state[7] = Driver Command

    modelParam.g 
    modelParam.frontTyre.B
    modelParam.frontTyre.C
    modelParam.frontTyre.D
    modelParam.rearTyre.B
    modelParam.rearTyre.C
    modelParam.rearTyre.D
    modelParam.C_m 
    modelParam.C_r0 
    modelParam.C_r2 
    modelParam.L_forward 
    modelParam.L_rear
    modelParam.mass 
    modelParam.I_z 
    modelParam.P_tv

    FN_r = ((modelParam.L_rear * modelParam.mass * modelParam.g) / (modelParam.L_forward + modelParam.L_rear))
    FN_f = ((modelParam.L_forward * modelParam.mass * modelParam.g) / (modelParam.L_forward + modelParam.L_rear))


    Fx = (modelParam.C_m*state[7]) - modelParam.C_r0 - (modelParam.C_r2*(state[3]**2))
    alpha_f = -np.atan((state[4] + modelParam.L_forward*state[5])/state[3]) + state[6]
    Fyf = FN_f*modelParam.frontTyre.D*np.sin(modelParam.frontTyre.C*np.atan(modelParam.frontTyre.B*alpha_f))
    alpha_r = -np.atan((state[4] - modelParam.L_rear*state[5])/state[3]) 
    Fyr = FN_r*modelParam.rearTyre.D*np.sin(modelParam.rearTyre.C*np.atan(modelParam.rearTyre.B*alpha_r))
    Mtv = modelParam.P_tv*(((np.tan(state[6])*state[3])/(modelParam.L_rear + modelParam.L_forward)) - state[5])

    sdot = ((state[3]*np.cos(state[2])) - (state[4]*np.sin(state[2])))/(1 - state[1]*curvature)
    ndot = (state[3]*np.sin(state[2])) + (state[4]*np.cos(state[2]))
    mudot = state[5] - (curvature*sdot)
    vxdot = (1/modelParam.mass)*(Fx - Fyf*np.sin(state[6]) + modelParam.mass*state[4]*state[5])
    vydot = (1/modelParam.mass)*(Fyr + Fyr*np.cos(state[6]) - modelParam.mass*state[3]*state[5])
    rdot = (1/modelParam.I_z) * (Fyf*modelParam.L_forward*np.cos(state[6]) - Fyr*modelParam.L_rear + Mtv)
    steer_dot = input[0]
    T_dot = input[1]
    return (sdot, ndot, mudot, vxdot, vydot, rdot, steer_dot, T_dot)

def state_approximation():
    pass

def coordinate_transform():
    pass

def local_planner(current_state):
    pass





# Over arching functions

def update_car_dynamics():
    pass
def car_control():
    pass
def update_local_plan():
    pass

# Environment loop
for step in range((maxtime/timestep)):
    current_time = step * timestep
    if current_time == maxtime:
        break

    update_car_dynamics() #Updates and keeps track of the sim Car dynamics
    
    if((current_time % sample_time) == 0):
        car_control() # At every sampling time, give a input to controls of the car
    if((current_time % update_time) == 0):
        update_local_plan() # At every update time, rerun the local planner, and recreate the local plan






