import numpy as np
import pandas as pd

class Obstacle:
    def __init__(self, N, delta_S, start_s, start_index, end_s, end_index, widths):
        self.N = N
        self.delta_s = delta_S
        self.start_s = start_s
        self.start_index = start_index
        self.end_s = end_s
        self.end_index = end_index
        self.widths = widths
    


class Track: 
    def __init__(self, N, delta_S, Curvature=None, Width=None, Track_Points=None, Yaw_Angle=None, Reference_Angle=None, Gradient=None, dx_perS=None, dy_perS=None, map_image=None, map_height=None, map_width=None, map_resolution=None,  Name=None, Origin=None, Obstacles=None):
        if (type(N) == int):
            self.N = N
        else:
            raise Exception("Incorrect argument for N")  
        
        if  (type(delta_S) == int or type(delta_S) == float):
            self.delta_s = delta_S
        else:
            raise Exception("Incorrect argument for Delta S")

        if Name is not None: 
            if type(Name) == str:
                self.Name = Name
            else:
                raise Exception("Incorrect argument for Name")
        else:
            self.Name = None

        if (Curvature is not None) and (Width is not None):
            self.new_track(Curvature, Width, Track_Length=self.N)

        self.track_points = Track_Points
        self.yaw_angle = Yaw_Angle
        self.reference_angle = Reference_Angle
        self.gradient = Gradient
        self.dx = dx_perS
        self.dy = dy_perS
        self.map_image = map_image
        self.map_height = map_height
        self.map_width = map_width
        self.map_resolution = map_resolution
        self.origin = Origin
        self.obstacles = Obstacles
        # self.obstacles = []
        # if Obstacles is not None:
        #     if isinstance(Obstacles, list):
        #         for ob in Obstacles:
        #             self.obstacles.append(ob)
        #     else: 
        #         self.obstacles.append(Obstacles)

    def set_constant_curvature(self, Curvature, Start=None, Stop=None):
        if (type(Curvature) == int or type(Curvature) == float):
            raise Exception("Incompatible Arguments")
        if Start is None:
            for i in range(self.N):
                self.curvature = Curvature
        elif Stop is None:
            if (type(Start) != int):
                raise Exception("Incompatible Argument for Start")
            for i in range(Start, self.N):
                self.curvature = Curvature
        else:
            if (type(Start) != int) or (type(Stop) != int) :
                raise Exception("Incompatible Argument for Start")
            for i in range(Start, Stop):
                self.curvature = Curvature

    def set_constant_width(self, Width, Start=None, Stop=None):
        if (type(Width) == int or type(Width) == float):
            raise Exception("Incompatible Arguments")
        if Start is None:
            for i in range(self.N):
                self.width = Width
        elif Stop is None:
            if (type(Start) != int):
                raise Exception("Incompatible Argument for Start")
            for i in range(Start, self.N):
                self.width = Width
        else:
            if (type(Start) != int) or (type(Stop) != int) :
                raise Exception("Incompatible Argument for Start")
            for i in range(Start, Stop):
                self.width = Width

    def new_track(self, Curvature, Width, Track_Length=None):
        if not ((hasattr(Curvature, '__len__') or (type(Curvature) == int or type(Curvature) == float)) and (hasattr(Width, '__len__') or (type(Width) == int or type(Width) == float))):
            raise Exception("Incompatible Arguments Given for Cuvature and Widths")
        
        if Track_Length is None:
            if (type(Curvature) == int or type(Curvature) == float) and (type(Width) == int or type(Width) == float):
                raise Exception("No Track length Given")
            
            if hasattr(Curvature, '__len__') and hasattr(Width, '__len__'):
                if len(Curvature) != len(Width):
                    raise Exception("Inconsistent Track length between Curvature and Width")
                self.N = len(Curvature)
                self.curvature = np.zeros(self.N)
                self.width = np.zeros(self.N)
                for i in range(self.N):
                    self.curvature[i] = Curvature[i]
                    self.width[i] = Width[i]
            
            elif hasattr(Curvature, '__len__') and (type(Width) == int or type(Width) == float):
                self.N = len(Curvature)
                self.curvature = np.zeros(self.N)
                self.width = np.zeros(self.N)
                for i in range(self.N):
                    self.curvature[i] = Curvature[i]
                    self.width[i] = Width
            
            elif  (type(Curvature) == int or type(Curvature) == float) and hasattr(Width, '__len__'):
                self.N = len(Width)
                self.curvature = np.zeros(self.N)
                self.width = np.zeros(self.N)
                for i in range(self.N):
                    self.curvature[i] = Curvature
                    self.width[i] = Width[i]
            else:
                raise Exception("Incompatible Argument Types")
     
        if Track_Length is not None:
            if (type(Track_Length) == int):
                if hasattr(Curvature, '__len__'):
                    if len(Curvature) != Track_Length:
                        raise Exception("Inconsistent Track length between Curvature and Track_Length")
                if hasattr(Width, '__len__'):
                    if len(Width) != Track_Length:
                        raise Exception("Inconsistent Track length between Width and Track_Length")
                if (type(Curvature) == int or type(Curvature) == float) and (type(Width) == int or type(Width) == float):
                    self.N = Track_Length
                    self.curvature = np.zeros(self.N)
                    self.width = np.zeros(self.N)
                    for i in range(self.N):
                        self.curvature[i] = Curvature
                        self.width[i] = Width

                elif hasattr(Curvature, '__len__') and hasattr(Width, '__len__'):
                    if len(Curvature) != len(Width):
                        raise Exception("Inconsistent Track length between Curvature and Width")
                    self.N = Track_Length
                    self.curvature = np.zeros(self.N)
                    self.width = np.zeros(self.N)
                    for i in range(self.N):
                        self.curvature[i] = Curvature[i]
                        self.width[i] = Width[i]
            
                elif hasattr(Curvature, '__len__') and (type(Width) == int or type(Width) == float):
                    self.N = Track_Length
                    self.curvature = np.zeros(self.N)
                    self.width = np.zeros(self.N)
                    for i in range(self.N):
                        self.curvature[i] = Curvature[i]
                        self.width[i] = Width
            
                elif  (type(Curvature) == int or type(Curvature) == float) and hasattr(Width, '__len__'):
                    self.N = Track_Length
                    self.curvature = np.zeros(self.N)
                    self.width = np.zeros(self.N)
                    for i in range(self.N):
                        self.curvature[i] = Curvature
                        self.width[i] = Width[i]
                else:
                    raise Exception("Incompatible Argument Types")
            else:
                raise Exception("Incorrect format for Track_Length")
    
    def exportTrackParameters(self, path):
        line1 = "Number of Track discretizations: " + str(self.N) + "\n"
        line2 = "Discretization Distance: " + str(self.delta_s) + "\n"
        if self.Name is not None:
            if type(self.Name) == str:
                line3 = "Track name: " + str(self.Name) + "\n"
        else: 
            line3 = "Track name: No Name Given\n"
        temp_path = path + "/Track Information.txt" 
        f = open(temp_path, "w")
        f.write(line1)
        f.write(line2)
        f.write(line3)
        f.close()
        df = pd.DataFrame({"Curvature" : self.curvature, "Width" : self.width})
        csv_path = path + "/Track Information.csv" 
        df.to_csv(csv_path, index=False)
