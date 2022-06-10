import numpy as np
from mapping import map
from optimization import Track
from optimization import Model_Parameters
from optimization import Results
from optimization import Optimizer

class auto_car:
    
    def __init__(self, global_track):
        self.global_track = global_track

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