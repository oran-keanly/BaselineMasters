
import sys
import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import image
from curv import Curvature
from curv import GradientCurvature
import math
import cmath
import yaml
from argparse import Namespace
from PIL import Image, ImageOps, ImageDraw
import functions
from scipy import ndimage 
import cubic_spline_planner
import optimization as opt

class map:
    
    def __init__(self, map_name):
        self.map_name = map_name
        self.read_yaml_file()
        self.load_map()
        self.occupancy_grid_generator()

    def find_curvature(self, X, Y):
        track_xy = list(zip(X, Y))
        curv1 = GradientCurvature(trace=track_xy, interpolate=False)
        #curv1.calculate_curvature()
        curv2 = Curvature(trace=track_xy)#, interpolate=True, plot_derivatives=True)
        #curv2.calculate_curvature()
        
        self.curvature = curv1.calculate_curvature()
        
        return self.curvature
    
    def read_yaml_file(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            yaml_file = dict(yaml.full_load(file).items())

        self.resolution = yaml_file['resolution']
        self.origin = yaml_file['origin']
        self.map_img_name = yaml_file['image']
    
    def load_map(self):
        image_path = sys.path[0] + '/maps/' + self.map_name + '.png'
        with Image.open(image_path) as im:
            self.gray_im = ImageOps.grayscale(im)
        self.map_height = self.gray_im.height*self.resolution
        self.map_width = self.gray_im.width*self.resolution
            
    def occupancy_grid_generator(self):  
        self.map_array = np.asarray(self.gray_im)
        self.occupancy_grid = self.map_array<1
    
    def set_true_widths(self):
        tx = self.cline[:, 0]
        ty = self.cline[:, 1]

        sf = 0.9 # safety factor
        nws, pws = [], []
        for i in range(self.N):  
            pt = [tx[i], ty[i]]
            c, r = self.xy_to_row_column(pt)
            val = self.dt[r, c] * sf
            nws.append(val)
            pws.append(val)

        nws, pws = np.array(nws), np.array(pws)
        
        self.widths =  np.concatenate([nws[:, None], pws[:, None]], axis=-1)     
        # self.widths *= 0.2 #TODO: remove
        self.widths *= 0.7 #TODO: remove
        
    def find_centerline(self, show=False):
        
        #self.dt = ndimage.distance_transform_edt(self.gray_im) 
        #self.dt = np.array(self.dt *self.resolution)
        self.dt = ndimage.distance_transform_edt(np.flipud(np.invert(self.occupancy_grid)))
        self.dt = np.array(self.dt *self.resolution)
        dt = self.dt

        d_search = 0.8
        n_search = 20
        dth = (np.pi * 4/5) / (n_search-1)

        # makes a list of search locations
        search_list = []
        for i in range(n_search):
            th = -np.pi/2 + dth * i
            x = -np.sin(th) * d_search
            y = np.cos(th) * d_search
            loc = [x, y]
            search_list.append(loc)

        pt = start = np.array([0, 0]) #TODO: start from map position
        self.cline = [pt]
        self.cline_row_coloumn = [pt]
        #th = self.stheta
        th = 0

        while (functions.get_distance(pt, start) > d_search/2 or len(self.cline) < 10) and len(self.cline) < 500:
            vals = []
            self.search_space = []
            for i in range(n_search):
                d_loc = functions.transform_coords(search_list[i], -th)
                search_loc = functions.add_locations(pt, d_loc)

                self.search_space.append(search_loc)

                x, y = self.xy_to_row_column(search_loc)
                val = dt[y, x]
                vals.append(val)

            ind = np.argmax(vals)
            d_loc = functions.transform_coords(search_list[ind], -th)
            pt = functions.add_locations(pt, d_loc)
            self.cline.append(pt)
            self.cline_row_coloumn.append(self.xy_to_row_column(pt))

            if show:
                self.plot_raceline_finding()

            th = functions.get_bearing(self.cline[-2], pt)
            #print(f"Adding pt: {pt}")

        self.cline = np.array(self.cline)
        self.cline_row_coloumn = np.array(self.cline_row_coloumn)
        self.N = len(self.cline)
        #print(f"Raceline found --> n: {len(self.cline)}")
        if show:
            self.plot_raceline_finding(True)
        #self.plot_raceline_finding(False)

        self.centerline = np.array(self.cline)
        self.centerline[:,0] = self.centerline[:,0]-self.origin[0]
        self.centerline[:,1] = self.centerline[:,1]-self.origin[1]
        
        self.centerline = self.centerline[1:-1,:]
        self.centerline = np.reshape(np.append(self.centerline, self.centerline[0,:]), (int(len(np.append(self.centerline, self.centerline[0,:]))/2), 2)) 
        
        #plt.imshow(self.gray_im, extent=(0,self.map_width,0,self.map_height))
        #plt.plot(self.centerline[:,0], self.centerline[:,1], 'x')
        #plt.show()

    def plot_raceline_finding(self, wait=False):
        plt.figure(1)
        plt.clf()
        plt.imshow(self.dt, origin='lower')

        for pt in self.cline:
            s_x, s_y = self.xy_to_row_column(pt)
            plt.plot(s_x, s_y, '+', markersize=16)

        for pt in self.search_space:
            s_x, s_y = self.xy_to_row_column(pt)
            plt.plot(s_x, s_y, 'x', markersize=12)


        plt.pause(0.001)

        if wait:
            plt.show()

    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        if c >= self.dt.shape[1]:
            c = self.dt.shape[1] - 1
        if r >= self.dt.shape[0]:
            r = self.dt.shape[0] - 1

        return c, r
    
    #def generate_line(self)

    def generate_track(self):
        ds = 0.1
        self.find_centerline(True)

        self.cline_row_coloumn[0, :] = self.cline_row_coloumn[-1, :]
        self.centerline[0, :] = self.centerline[-1, :]

        track_coords = self.cline_row_coloumn[:, 0:2] * self.resolution 

        #rx, ry, ryaw, rk, d = cubic_spline_planner.calc_spline_course(self.cline_row_coloumn[:,0], self.cline_row_coloumn[:,1], ds)
        rx, ry, ryaw, rk, d = cubic_spline_planner.calc_spline_course(track_coords[:,0], track_coords[:,1], ds)
        self.track_pnts = np.array((rx,ry))
        self.curvature = rk
        self.gradients = ryaw # in rad
        for i in range(len(rx)):
            temp = np.arctan(ry[i]/rx[i])
            print("Returned yaw is: {}\n" .format(np.rad2deg(ryaw[i])))
            print("Angle from x axis is yaw is: {}\n\n" .format(np.rad2deg(temp)))
        #print(self.track_pnts.shape)
        self.find_curvature(rx, ry)
        plt.close()
        plt.plot(self.curvature)
        plt.show()
        self.set_true_widths() #TODO: IMPOROVE WIDTH FUNCTION

        #plt.imshow(self.gray_im, extent=(0,(self.map_width/self.resolution),0,(self.map_height/self.resolution)))
        plt.plot(rx, ry)
        #plt.plot(self.cline_row_coloumn[:,0], self.cline_row_coloumn[:,1], 'x')
        plt.plot(self.cline_row_coloumn[:,0]* self.resolution , self.cline_row_coloumn[:,1]* self.resolution , 'x')
        plt.show()

        #self.track = opt.Track(len(rx), ds, self.curvature, 0.8, self.map_name)
        N = 1000
        temp = np.arange(4000)
        curve = np.zeros(N)
        # for i in range(N):
        #     curve[i] = 0.04 * np.sin(i * 2*np.pi * (1/N))
        # plt.close()
        # plt.plot(temp, curve)
        # plt.show()
        self.track = opt.Track(N, ds, curve, 0.9, "Track Test")

    def find_trajectory(self):
        self.generate_track()
        optimise = opt.Optimizer(Track=self.track)
        
        delta_st = 0.0
        delta_t = 0.0
        n = 0.0
        mu = 0.2
        vx = 5.0
        vy = 0.0
        r = 0.0
        st = 0.0
        t = 1.0
        initcondition = opt.State_parameters(delta_st, delta_t, n, mu, vx, vy, r, st, t)
        m_param = opt.Model_Parameters( Max_Tyre_Force=500.0, Slip_AngleReg=20.0, Steer_Reg=5.0, Drive_Reg=0.0)

        results = optimise.optimize(initcondition, Parameters=m_param, finalCondition=None, Temp_Track=None)
        self.results = results
        

if __name__=='__main__':
    #test_map()
    #m = map('porto_1') #Doesn't Work
    #m = map('circle') #Doesn't Work
    #m = map('berlin') #Doesn't Work
    #m = map('f1_aut_wide') #Doesn't Work
    m = map('columbia_1') #Works
    m.generate_track()
    #m.find_trajectory()