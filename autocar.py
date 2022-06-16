import numpy as np
from mapping import map
import LibFunctions as lib
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
            self.initialConditions = State_parameters(0, 0, 0, 0, 5, 0, 0, 0, 1) # my Model; 
        else:
            self.initialConditions = initialConditions
        self.model_parameters = Model_Parameters(Slip_AngleReg=5, Steer_Reg=5)
        opt = Optimizer()
        self.Global_Reference = opt.optimize(self.initialConditions, Parameters=self.model_parameters, Temp_Track=self.global_track)
        self.plot_global_reference()

    def linearly_interpolate(self, yA, yB, xA, xB, xC):
        gradient = (yB - yA)/(xB - xA)
        diff = xC - xA
        value = (gradient*diff) + yA
        return value

    def track_curvarture(self, s):
        for i in range((self.global_track.N)):
            if ( (self.global_track.delta_s * i ) <= s) and ( (self.global_track.delta_s * (i+1) ) >= s):
                curvature = self.linearly_interpolate((self.global_track.delta_s * i ), (self.global_track.delta_s * (i+1) ), s, self.global_track.delta_s)
                return curvature
            
        raise Exception("Unable to find track curvature at current position")

    def find_angle(self, A, B, C):
        # RETURNS THE ANGLE BÂC
        vec_AB = A - B
        vec_AC = A - C 
        dot = vec_AB.dot(vec_AC)
        #dot = (A[0] - C[0])*(A[0] - B[0]) + (A[1] - C[1])*(A[1] - B[1])
        magnitude_AB = np.linalg.norm(vec_AB)
        magnitude_AC = np.linalg.norm(vec_AC)

        angle = np.arccos(dot/(magnitude_AB*magnitude_AC))
        return angle

    def transform_XY_to_NS(self, x, y):
        distances = np.zeros(len(self.global_track.N))
        for i in range(len(self.global_track.N)):
            distances[i] = lib.get_distance(self.global_track.track_points[:, i], np.array[x, y])
        
        i_s = np.argmax(distances)
        if i_s >= 1:
            if distances[i_s-1] < distances[i_s+1]:
                i_s2 = i_s - 1
                ang = self.find_angle(self.global_track.track_points[:, i_s2], self.global_track.track_points[:, i_s], np.array[x, y])
                actual_s = (i_s2 * self.global_track.delta_s) + (np.cos(ang) * distances[i_s2])
                actual_n = (np.sin(ang) * distances[i_s2])
            elif  distances[i_s-1] > distances[i_s+1]:
                i_s2 = i_s + 1
                ang = self.find_angle(self.global_track.track_points[:, i_s], self.global_track.track_points[:, i_s2], np.array[x, y])
                actual_s = (i_s * self.global_track.delta_s) + (np.cos(ang) * distances[i_s])
                actual_n = (np.sin(ang) * distances[i_s])
            else:
                raise Exception("Error")
        else:
            i_s2 = i_s + 1
            ang = self.find_angle(self.global_track.track_points[:, i_s], self.global_track.track_points[:, i_s2], np.array[x, y])
            actual_s = (i_s * self.global_track.delta_s) + (np.cos(ang) * distances[i_s])
            actual_n = (np.sin(ang) * distances[i_s])
        return (actual_s, actual_n)   

    def transform_NS_to_XY(self, s, n):
        for i in range((self.global_track.N-1)):
            if ( (self.global_track.delta_s * i ) <= s) and ( (self.global_track.delta_s * (i+1) ) >= s):
                yaw_angle = self.linearly_interpolate(self.global_track.yaw_angle[i], self.global_track.yaw_angle[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                track_grad = self.linearly_interpolate(self.global_track.gradient[i], self.global_track.gradient[i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                track_x = self.linearly_interpolate(self.global_track.track_points[0, i], self.global_track.track_points[0, i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                track_y = self.linearly_interpolate(self.global_track.track_points[1, i], self.global_track.track_points[1, i+1], (self.global_track.delta_s*i), (self.global_track.delta_s*(i+1)), s)
                break
        actual_X = -n*np.sin(yaw_angle) + track_x
        actual_Y = n*np.cos(yaw_angle) + track_y

        return (actual_X, actual_Y)

    def plot_global_reference(self):
        optimal_xy_path = np.zeros((2, self.global_track.N))
        for i in range(self.global_track.N):
            x, y = self.transform_NS_to_XY(i*self.global_track.delta_s, self.Global_Reference.n[i])
            optimal_xy_path[0, i] = x
            optimal_xy_path[1, i] = y

        #plt.imshow(self.global_track.map_image, extent=(0,(self.global_track.map_width/self.global_track.map_resolution),0,(self.global_track.map_height/self.global_track.map_resolution)))
        #plt.imshow(self.global_track.map_image, extent=(0,(self.global_track.map_width/self.global_track.map_resolution),0,(self.global_track.map_height/self.global_track.map_resolution)))
        #plt.imshow(self.global_track.map_image, extent=(0,(self.global_track.map_width),0,(self.global_track.map_height)))
        #plt.plot(optimal_xy_path[0, :] * (1/self.global_track.map_resolution), optimal_xy_path[1, :] * (1/self.global_track.map_resolution))
        #plt.plot(optimal_xy_path[0, :], optimal_xy_path[1, :])
        plt.imshow(self.global_track.map_image, extent=(0,(self.global_track.map_width),0,(self.global_track.map_height)))
        #plt.plot(self.global_track.track_points[0,:] , self.global_track.track_points[1,:] )
        plt.plot(optimal_xy_path[0,:] , optimal_xy_path[1,:] )
        plt.show()
        self.Global_Reference_XY =  optimal_xy_path
    
    def car_control(self, state, time):
        return np.array([0, 1])
        #pass

    def update_local_plan(self, state):
        pass

    def approximate_lidar_scan(self):
        pass

    def state_approximation(self):
        pass

if __name__ == "__main__":
    m = map('columbia_1') #Works
    track = m.generate_track(0.1)
    # N = 500
    # curv = np.zeros(501)
    # for i in range(100, 201):
    #     curv[i] = (0.01/100)*i - 0.01
    # for i in range(200, 401):
    #     curv[i] = (-0.01/100)*i + 0.03
    # for i in range(400, 501):
    #     curv[i] = (0.01/100)*i - 0.05
    # # for i in range(N):
    # #     curv[i] = 0.01 * np.sin(i * 2*np.pi * (1/N))
    # plt.plot(curv)
    # plt.show()
    # track = Track(501, 0.1, curv, 6.0, Name="Attempt")
    agent = auto_car(track)
    #sim_env.main(agent)

