import numpy as np
from mapping import map
from matplotlib import  pyplot as plt
from optimization import State_parameters
from optimization import Track
from optimization import Model_Parameters
from optimization import Results
from optimization import Optimizer

class auto_car:
    
    def __init__(self, global_track, initialConditions=None):
        self.global_track = global_track
        if initialConditions is None:
            self.initialConditions = State_parameters(0, 0, 0, 0, 0.005, 0, 0, 0, 0) # my Model; 
        else:
            self.initialConditions = initialConditions
        self.model_parameters = Model_Parameters(Slip_AngleReg=5, Steer_Reg=5)
        opt = Optimizer()
        self.Global_Reference = opt.optimize(self.initialConditions, Parameters=self.model_parameters, Temp_Track=self.global_track)

    
    def car_control(self, state, time):
        return np.array([0, 1])
        #pass

    def update_local_plan(self, state):
        pass

    def linearly_interpolate(self, point1, point2, interpolate_point, spacing):
        gradient = (point2 - point1)/spacing
        diff = interpolate_point - point1
        value = (gradient*diff) + point1
        return value

    def approximate_lidar_scan(self):
        pass

    def state_approximation(self):
        pass

if __name__ == "__main__":
    #m = map('columbia_1') #Works
    #track = m.generate_track(0.1)
    curv = np.zeros(500)
    temp = 0
    for i in range(100, 201):
        curv[i] = (0.01/100)*i - 0.01
    for i in range(200, 301):
        curv[i] = 0.01
    for i in range(300, 401):
        curv[i] = -(0.01/100)*i + 0.04
    plt.plot(curv)
    plt.show()
    track = Track(500, 0.1, curv, 1, Name="Attempt")
    agent = auto_car(track)
    #sim_env.main(agent)

