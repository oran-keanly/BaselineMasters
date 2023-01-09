
import sys
import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import image
import math
import cmath
import yaml
from argparse import Namespace
from PIL import Image, ImageOps, ImageDraw
import functions
from scipy import ndimage 
import cubic_spline_planner
from track import Track

class map:
    
    def __init__(self, map_name):
        self.map_name = map_name
        self.read_yaml_file()
        self.load_map()
        self.occupancy_grid_generator()
    
    def read_yaml_file(self):
        #file_name = 'maps/' + self.map_name + '.yaml'
        #file_name = self.map_name + '.yaml'
        file_name = f'/home/oran/catkin_ws/src/F1tenth/ros1/f1tenth_gym_ros/maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            yaml_file = dict(yaml.full_load(file).items())

        self.resolution = yaml_file['resolution']
        self.origin = yaml_file['origin']
        self.map_img_name = yaml_file['image']
    
    def load_map(self):
        #image_path = sys.path[0] + '/maps/' + self.map_name + '.png'
        image_path = f'/home/oran/catkin_ws/src/F1tenth/ros1/f1tenth_gym_ros/maps/' + self.map_name + '.png'
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

        #d_search = 0.3 # Berlin
        d_search = 0.8 # Columbia
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
    
    def generate_track(self, delta_s, width=None):
        ds = delta_s
        self.find_centerline(False)

        self.cline_row_coloumn[0, :] = self.cline_row_coloumn[-1, :]
        self.centerline[0, :] = self.centerline[-1, :]

        track_coords = self.cline_row_coloumn[:, 0:2] * self.resolution 

        #rx, ry, ryaw, rk, dx, ddx, dy, ddy, s = cubic_spline_planner.calc_spline_course(self.cline_row_coloumn[:,0], self.cline_row_coloumn[:,1], ds)
        rx, ry, ryaw, rk, dx, ddx, dy, ddy, s= cubic_spline_planner.calc_spline_course(track_coords[:,0], track_coords[:,1], ds)
        self.N = len(rx)
        self.track_pnts = np.array((rx,ry))
        self.curvature = np.array(rk)
        # with np.printoptions(threshold=np.inf):
        #     print(self.track_pnts)
        # plt.close()
        # plt.plot(self.curvature * 0.3)
        # plt.show()
        #self.yaw = ryaw # in rad
        for i in range(len(ryaw)-1):
            if np.abs(ryaw[i] - ryaw[i+1]) >= (np.pi*1.5):
                
                if ryaw[i] > ryaw[i+1]:
                    while np.abs(ryaw[i] - ryaw[i+1]) >= (np.pi*1.5):
                        ryaw[i+1] += 2*np.pi
                    
                if ryaw[i] < ryaw[i+1]:
                    while np.abs(ryaw[i] - ryaw[i+1]) >= (np.pi*1.5):
                        ryaw[i+1] -= 2*np.pi

        self.yaw = ryaw
        # plt.plot(np.rad2deg(self.yaw))
        # plt.show()
        # with np.printoptions(threshold=np.inf):
        #     print(np.rad2deg(self.yaw))
        self.ref_angle = np.zeros(self.N)
        self.gradient = np.zeros(self.N)
        for i in range(len(rx)):
            self.ref_angle[i] = np.arctan(ry[i]/rx[i])
            if(dx[i] != 0):
                self.gradient[i] = dy[i]/(1/dx[i])
            else:
                self.gradient[i] = 1000000


        #self.set_true_widths() #TODO: IMPOROVE WIDTH FUNCTION
        #self.widths = 0.3 # ESL TEST TRACK
        if width is None:
            self.widths = 1.4# COLUMBIA_1 TRACK
        else:    
            self.widths = width
        #self.widths = 1.1 # Berlin TRACK

        self.track = Track(self.N, ds, Curvature=self.curvature, Width=self.widths, Track_Points=self.track_pnts, Yaw_Angle=self.yaw, Reference_Angle=self.ref_angle, Gradient=self.gradient, dx_perS=dx, dy_perS=dy, map_image=self.gray_im, map_height=self.map_height, map_width=self.map_width, map_resolution=self.resolution,  Name= self.map_name, Origin=self.origin)
        # plt.imshow(self.gray_im, extent=(0,(self.map_width),0,(self.map_height)))
        # plt.plot(self.track_pnts[0,:] , self.track_pnts[1,:]  )
        # plt.plot(self.cline_row_coloumn[:,0]* self.resolution , self.cline_row_coloumn[:,1]* self.resolution , 'x')
        # plt.show()

        return self.track
        

if __name__=='__main__':
    #test_map()
    #m = map('porto_1') #Doesn't Work
    #m = map('circle') #Doesn't Work
    #m = map('berlin') 
    #m = map('f1_aut_wide') #Doesn't Work
    m = map('columbia_small') #Works
    # m = map('test_ESL_map')
    delta_s = 1#0.1
    m.generate_track(delta_s)
    #m.find_trajectory()