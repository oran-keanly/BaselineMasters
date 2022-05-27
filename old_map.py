from numpy.core.fromnumeric import shape
from curv import Curvature
from curv import GradientCurvature
import yaml 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import csv
#import casadi as ca 
from scipy import ndimage 
from scipy.interpolate import UnivariateSpline
import io
import LibFunctions as lib
import optimization as opt
#import optimization as opt



class PreMap:
    def __init__(self, conf, map_name) -> None:
        self.conf = conf #TODO: update to use new config style
        self.map_name = map_name

        self.map_img = None
        self.origin = None
        self.resolution = None
        self.stheta = None
        self.map_img_name = None

        self.cline = None
        self.nvecs = None
        self.widths = None

        self.wpts = None
        self.vs = None

        self.myTrack = None

    def run_conversion(self):
        self.read_yaml_file()
        self.load_map()

        self.dt = ndimage.distance_transform_edt(self.map_img) 
        self.dt = np.array(self.dt *self.resolution)
        self.find_centerline(True)
        self.find_nvecs_old()
        #self.find_nvecs()
        self.set_true_widths()
        self.render_map()

        self.save_map_std()
        self.save_map_centerline_oran()
        #self.run_optimisation_no_obs()
        #self.save_map_opti()

        self.render_map(True)

    def run_opti(self):
        self.read_yaml_file()
        self.load_map()

        self.dt = ndimage.distance_transform_edt(self.map_img) 
        self.dt = np.array(self.dt *self.resolution)

        self.run_optimisation_no_obs()
        self.save_map_opti()

    def load_track_pts(self):
        track = []
        filename = 'maps/' + self.name + "_std.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename} in env_map")

        self.N = len(track)
        self.track_pts = track[:, 0:2]
        self.nvecs = track[:, 2: 4]
        self.ws = track[:, 4:6]
        
    def read_yaml_file(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

        yaml_file = dict(documents.items())

        self.resolution = yaml_file['resolution']
        self.origin = yaml_file['origin']
        self.stheta = yaml_file['start_pose'][2]
        self.map_img_name = yaml_file['image']

    def load_map(self):
        map_img_name = 'maps/' + self.map_img_name

        try:
            self.map_img = np.array(Image.open(map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        except Exception as e:
            print(f"MapPath: {map_img_name}")
            print(f"Exception in reading: {e}")
            raise ImportError(f"Cannot read map")
        try:
            self.map_img = self.map_img[:, :, 0]
        except:
            pass

        self.height = self.map_img.shape[1]
        self.width = self.map_img.shape[0]

    def find_curvature(self, X, Y):
        track_xy = list(zip(X, Y))
        curv1 = GradientCurvature(trace=track_xy, interpolate=False)
        #curv1.calculate_curvature()
        curv2 = Curvature(trace=track_xy)#, interpolate=True, plot_derivatives=True)
        #curv2.calculate_curvature()
        
        self.Curvature = curv1.calculate_curvature()
        
        return self.Curvature

    def find_centerline(self, show=True):
        dt = self.dt
  
        d_search = 0.2
        n_search = 11
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
        self.myTrack = [pt]
        th = self.stheta
        while (lib.get_distance(pt, start) > d_search/2 or len(self.cline) < 10) and len(self.cline) < 500:
            vals = []
            self.search_space = []
            for i in range(n_search):
                d_loc = lib.transform_coords(search_list[i], -th)
                search_loc = lib.add_locations(pt, d_loc)

                self.search_space.append(search_loc)

                x, y = self.xy_to_row_column(search_loc)
                val = dt[y, x]
                vals.append(val)

            ind = np.argmax(vals)
            d_loc = lib.transform_coords(search_list[ind], -th)
            pt = lib.add_locations(pt, d_loc)
            self.cline.append(pt)
            self.myTrack.append(self.xy_to_row_column(pt))
            if show:
                self.plot_raceline_finding()

            th = lib.get_bearing(self.cline[-2], pt)
            print(f"Adding pt: {pt} and co-ordinate {self.xy_to_row_column(pt)}")

        self.cline = np.array(self.cline)
        self.myTrack = np.array(self.myTrack)
        self.N = len(self.cline)
        print(f"Raceline found --> n: {len(self.cline)}")
        if show:
            self.plot_raceline_finding(True)
        self.plot_raceline_finding(False)

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

    def find_nvecs(self):
        N = len(self.cline)

        n_search = 64
        d_th = np.pi * 2 / n_search
        xs, ys = [], []
        for i in range(n_search):
            th = i * d_th
            xs.append(np.cos(th))
            ys.append(np.sin(th))

        xs = np.array(xs)
        ys = np.array(ys)

        sf = 0.8
        nvecs = []
        widths = []
        for i in range(self.N):
            pt = self.cline[i]
            c, r = self.xy_to_row_column(pt)
            val = self.dt[r, c] * sf 
            widths.append(val)

            s_vals = np.zeros(n_search)
            s_pts = np.zeros((n_search, 2))
            for j in range(n_search):
                dpt = np.array([xs[j]+val, ys[j]*val]) / self.resolution
                # dpt_c, dpt_r = self.xy_to_row_column(dpt)
                # s_vals[i] = self.dt[r+dpt_r, c+dpt_c]
                s_pt = [int(round(r+dpt[1])), int(round(c+dpt[0]))]
                s_pts[j] = s_pt
                s_vals[j] = self.dt[s_pt[0], s_pt[1]]

            print(f"S_vals: {s_vals}")
            idx = np.argmin(s_vals) # closest to border

            th = d_th * idx

            nvec = [xs[idx], ys[idx]]
            nvecs.append(nvec)

            self.plot_nvec_finding(nvecs, widths, s_pts, pt, True)

        self.nvecs = np.array(nvecs)
        plt.show()

    def find_nvecs_old(self):
        N = self.N
        track = self.cline

        nvecs = []
        # new_track.append(track[0, :])
        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[0, :], track[1, :]))
        nvecs.append(nvec)
        for i in range(1, len(track)-1):
            pt1 = track[i-1]
            pt2 = track[min((i, N)), :]
            pt3 = track[min((i+1, N-1)), :]

            th1 = lib.get_bearing(pt1, pt2)
            th2 = lib.get_bearing(pt2, pt3)
            if th1 == th2:
                th = th1
            else:
                dth = lib.sub_angles_complex(th1, th2) / 2
                th = lib.add_angles_complex(th2, dth)

            new_th = th + np.pi/2
            nvec = lib.theta_to_xy(new_th)
            nvecs.append(nvec)

        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[-2, :], track[-1, :]))
        nvecs.append(nvec)

        self.nvecs = np.array(nvecs)

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def plot_nvec_finding(self, nvecs, widths, s_pts, c_pt, wait=False):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.map_img, origin='lower')

        xs, ys = [], []
        for pt in self.cline:
            s_x, s_y = self.xy_to_row_column(pt)
            xs.append(s_x)
            ys.append(s_y)
        plt.plot(xs, ys, linewidth=2)

        for i in range(len(s_pts)-1):
            plt.plot(s_pts[i, 1], s_pts[i, 0], 'x')

        c, r = self.xy_to_row_column(c_pt)
        plt.plot(c, r, '+', markersize=20)

        for i in range(len(nvecs)):
            pt = self.cline[i]
            n = nvecs[i]
            w = widths[i]
            dpt = np.array([n[0]*w, n[1]*w])
            p1 = pt - dpt
            p2 = pt + dpt

            lx, ly = self.convert_positions(np.array([p1, p2]))
            plt.plot(lx, ly, linewidth=1)

            # plt.plot(p1, p2)
        plt.pause(0.001)


        cx, cy = self.convert_positions(self.cline)
        plt.plot(cx, cy, '--', linewidth=2)

        # plt.show()
        plt.pause(0.0001)
        if wait:
            plt.show()

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
        
    def render_map(self, wait=False):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.map_img, origin='lower')

        xs, ys = [], []
        for pt in self.cline:
            s_x, s_y = self.xy_to_row_column(pt)
            xs.append(s_x)
            ys.append(s_y)
        plt.plot(xs, ys, linewidth=2)

        ns = self.nvecs 
        ws = self.widths
        l_line = self.cline - np.array([ns[:, 0] * ws[:, 0], ns[:, 1] * ws[:, 0]]).T
        r_line = self.cline + np.array([ns[:, 0] * ws[:, 1], ns[:, 1] * ws[:, 1]]).T

        cx, cy = self.convert_positions(self.cline)
        plt.plot(cx, cy, '--', linewidth=2)
        lx, ly = self.convert_positions(l_line)
        plt.plot(lx, ly, linewidth=1)
        rx, ry = self.convert_positions(r_line)
        plt.plot(rx, ry, linewidth=1)

        if self.wpts is not None:
            wpt_x, wpt_y = self.convert_positions(self.wpts)
            plt.plot(wpt_x, wpt_y, linewidth=2)

        # plt.show()
        plt.pause(0.0001)
        if wait:
            plt.show()

    def check_scan_location(self, pt):
        c, r = self.xy_to_row_column(pt)
        if abs(c) > self.width -2 or abs(r) > self.height -2:
            return True
        val = self.dt[c, r]
        if val < 0.05:
            return True
        return False

    def save_map_std(self):
        filename = 'maps/' + self.map_name + '_std.csv'

        track = np.concatenate([self.cline, self.nvecs, self.widths], axis=-1)

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(track)

        print(f"Track Saved in File: {filename}")

    def save_map_centerline(self):
        filename = 'maps/' + self.map_name + '_centerline.csv'

        track = np.concatenate([self.cline, self.widths], axis=-1)
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(track)

        print(f"Track Saved in File: {filename}")

    def save_map_centerline_oran(self):
        filename = 'maps/' + self.map_name + '_centerline_oran.csv'

        track = np.concatenate([self.myTrack, self.widths], axis=-1)
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(track)

        print(f"Track Pixel Co-ordinates Saved in File: {filename}")

    def run_optimisation_no_obs(self):
        n_set = mincurvaturetrajectory(self.cline, self.nvecs, self.widths)

        deviation = np.array([self.nvecs[:, 0] * n_set[:, 0], self.nvecs[:, 1] * n_set[:, 0]]).t
        self.wpts = self.cline + deviation

        # self.vs = max_velocity(self.wpts, self.conf, false)
        self.vs = max_velocity(self.wpts, self.conf, true)

        plt.figure(4)
        plt.plot(self.vs)

    def save_map_opti(self):
        filename = 'maps/' + self.map_name + '_opti.csv'

        dss, ths = convert_pts_s_th(self.wpts)
        ss = np.cumsum(dss)
        ks = np.zeros_like(ths[:, None]) #TODO: add the curvature

        track = np.concatenate([ss[:, None], self.wpts[:-1], ths[:, None], ks, self.vs], axis=-1)

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(track)

        print(f"Track Saved in File: {filename}")

    def createTrack(self):
        self.find_centerline(show=False)
        Xref = self.myTrack[1:,1]
        Yref = self.myTrack[1:,0]
        curvature = self.find_curvature(Xref, Yref)
        print("Length of Track is: " + str(len(Xref)) + "\n")
        print("Length of Curvature is: " + str(len(curvature)) + "\n")
        plt.close()
        plt.plot(curvature)
        plt.show()
        delta_s = 0.2
        self.Track = opt.Track(len(Xref), delta_s, curvature, 0.8, self.map_name)
        self.set_true_widths()

    
def convert_pts_s_th(pts):
    N = len(pts)
    s_i = np.zeros(N-1)
    th_i = np.zeros(N-1)
    for i in range(N-1):
        s_i[i] = lib.get_distance(pts[i], pts[i+1])
        th_i[i] = lib.get_bearing(pts[i], pts[i+1])

    return s_i, th_i


def run_pre_map():
    fname = "config_test"
    conf = lib.load_conf(fname)
    # map_name = "example_map"
    #map_name = "columbia_small"
    map_name = "f1_aut_wide"
    

    pre_map = PreMap(conf, map_name)
    pre_map.run_conversion()
    pre_map.createTrack()
    # pre_map.run_opti()


if __name__ == "__main__":
    run_pre_map()