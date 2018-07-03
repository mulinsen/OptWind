# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gamma
import os
import re

class SiteCondition(object):
    """ Site condition contains mainly wind resource information for offshore
    and flat terrain sites.
    
    Parameters
    ----------
    Weibull_A: array:float
        
    Weibull_k: array:float
    
    frequency: array:float
    
    wd_binned: array:float, np.linspace(0, 330, num=12)
        Discretized wind direction bins at a reference height above the ground
        for far field inflow [deg]. This should be equally spaced and contains at
        least two wind directions, as we derive the sector width from it.
    
    height_ref: float, 85
        Reference height above the ground for defining the wind resource
        condition [m].
        
    z0: float, 0.001
        Roughness lenght, used to transfer wind speeds between different 
        heights above the ground [m].
    
    x_range: array:float or None
        Define the grid on which Weibull_A/k and frequency are defined, or None
        when the Weibull_A/k and frequency are uniform accross the site.
    
    y_range: array:float or None
        Same as x_range.

    """
    
    def __init__(self, Weibull_A, Weibull_k, frequency,
                 wd_binned=np.linspace(0, 330, num=12),
                 height_ref=85,
                 z0=0.001,
                 x_range=None,
                 y_range=None):
        
        self.Weibull_A = Weibull_A
        self.Weibull_k = Weibull_k
        self.frequency = frequency
        self.wd_binned = wd_binned
        self.wd_bin_size = wd_binned[1] - wd_binned[0]
        self.height_ref = height_ref
        self.x_range = x_range
        self.y_range = y_range
        self.uniform_flag = True
        
        ###################################################################
        # 1. Check agreement of inputs' shape and make sure frequency is -/deg
        # 1.1 uniform case
        if len(Weibull_A.shape) == 1:
            assert Weibull_A.shape[0] == len(wd_binned)
            
            if round(sum(self.frequency)) == 1:
                self.frequency /= self.wd_bin_size
            elif round(sum(self.frequency)) == 100:
                self.frequency /= 100
                self.frequency /= self.wd_bin_size
                
            self.mean_wind_speed = np.sum(
                    self.Weibull_A * gamma(1 + 1/self.Weibull_k) *
                    self.frequency * self.wd_bin_size)
            
        # 1. 2. non-uniform case
        elif len(Weibull_A.shape) == 3:
            assert (Weibull_A.shape[0] == len(x_range) and
                    Weibull_A.shape[1] == len(y_range) and
                    Weibull_A.shape[2] == len(wd_binned))
            
            self.uniform_flag = False
            
            if round(sum(self.frequency[0, 0, :])) == 1:
                self.frequency /= self.wd_bin_size
            elif round(sum(self.frequency[0, 0, :])) == 100:
                self.frequency /= 100
                self.frequency /= self.wd_bin_size
            
            self.mean_wind_speed = np.sum(
                    self.Weibull_A * gamma(1 + 1/self.Weibull_k) *
                    self.frequency * self.wd_bin_size, axis=-1)
        else:
            raise ValueError('The shape of Weibull_A is not correct!')
            
        
        ###################################################################
        # 2. Initialize interp funcs
        self.interp_funcs = {}
        wd_binned_extended = np.concatenate(
                (self.wd_binned, self.wd_binned[:1]+360))
        if self.uniform_flag:
            self.interp_funcs['Weibull_A'] = RegularGridInterpolator(
                    [wd_binned_extended],
                    np.concatenate((self.Weibull_A, self.Weibull_A[:1])),
                    bounds_error=False)
            
            self.interp_funcs['Weibull_k'] = RegularGridInterpolator(
                    [wd_binned_extended],
                    np.concatenate((self.Weibull_k, self.Weibull_k[:1])),
                    bounds_error=False)
            
            self.interp_funcs['frequency'] = RegularGridInterpolator(
                    [wd_binned_extended],
                    np.concatenate((self.frequency, self.frequency[:1])),
                    bounds_error=False)
        else:
            self.interp_funcs['Weibull_A'] = RegularGridInterpolator(
                    [x_range, y_range, wd_binned_extended],
                    np.concatenate((self.Weibull_A, self.Weibull_A[:, :, :1]), 
                                   axis=-1),
                    bounds_error=False)
            
            self.interp_funcs['Weibull_k'] = RegularGridInterpolator(
                    [x_range, y_range, wd_binned_extended],
                    np.concatenate((self.Weibull_k, self.Weibull_k[:, :, :1]), 
                                   axis=-1),
                    bounds_error=False)
            
            self.interp_funcs['frequency'] = RegularGridInterpolator(
                    [x_range, y_range, wd_binned_extended],
                    np.concatenate((self.frequency, self.frequency[:, :, :1]), 
                                   axis=-1),
                    bounds_error=False)
            
            self.interp_funcs['mean_wind_speed'] = RegularGridInterpolator(
                    [x_range, y_range], self.mean_wind_speed,
                    bounds_error=False)
            
            
    def getAkf(self, x=None, y=None, wd=None):
        """ Assume x, y and wd are float array with the same shape. """
        if wd is None:
            wd = self.wd_binned
        else:
            wd = np.mod(wd, 360)
            
        if self.uniform_flag:
            A = self.interp_funcs['Weibull_A']((wd)) 
            k = self.interp_funcs['Weibull_k']((wd)) 
            f = self.interp_funcs['frequency']((wd)) 
        else:
            A = self.interp_funcs['Weibull_A']((x, y, wd)) 
            k = self.interp_funcs['Weibull_k']((x, y, wd)) 
            f = self.interp_funcs['frequency']((x, y, wd)) 
        
        return A, k, f
    
    def get_mean_wind_speed(self, x=None, y=None):
        if self.uniform_flag:
            return self.mean_wind_speed
        else:
            return self.interp_funcs['mean_wind_speed']((x, y))
        
        
    def read_WAsP_grd(grd_file):
        
        num_head = 5
        # Read first 5 lines
        with open(grd_file) as grid:
            head = [next(grid).strip() for x in range(num_head+1)]
            
        (N_x, N_y) = [np.int(x) for x in head[1].split()]
        (X_min, X_max) = [np.float(x) for x in head[2].split()]
        (Y_min, Y_max) = [np.float(x) for x in head[3].split()]
        
        X_range = np.linspace(X_min, X_max, N_x)
        Y_range = np.linspace(Y_min, Y_max, N_y)
        
        # note there are two types of format, one with each row in one line,
        # the other with each row broken into multi-lines and seperated two 
        # rows with a blank line.
        if len(head[num_head].split()) == N_x:
            data = np.loadtxt(grd_file, skiprows=num_head)
        else:
            data = np.zeros((N_y, N_x))
            with open(grd_file) as grid:
                lines = grid.readlines()
            i_y = 0
            i_line = num_head
            
            data_row = []
            while i_y < N_y:
                data_row = [np.float(x) for x in lines[i_line].strip().split()]
                i_line = i_line + 1
                while len(lines[i_line].strip().split()) > 0:
                    data_row = data_row + [np.float(x) 
                               for x in lines[i_line].strip().split()]
                    i_line = i_line + 1
                    
                assert (len(data_row) == N_x)    
                data[i_y, :] = data_row
                
                i_line = i_line + 1
                i_y = i_y + 1
                            
        data = np.where(data == 1.70141E+038, np.nan, data)
        data = data.T #xy indexing to ij indexing
        
        return [X_range, Y_range, data]  
           
        
    @classmethod
    def from_WAsP_grd(cls, grd_file_path, z0=0.001):
        files = os.listdir(grd_file_path)
        Weibull_A_grds = []
        Weibull_k_grds = []
        frequency_grds = []
        
        first_match = True
        
        pattern = r'Sector (\d+) \s+ Height (\d+)m \s+ ([a-zA-Z0-9- ]+).grd'
        for file in files:
            match = re.findall(pattern, file)
            if len(match) == 1 and len(match[0]) == 3:
                match = match[0]
                i_sector = int(match[0]) - 1 # change into python convention
                height_ref_current = float(match[1])
                
                x_range_current, y_range_current, data = cls.read_WAsP_grd(
                        os.path.join(grd_file_path, file))
                if first_match:
                    x_range, y_range = x_range_current, y_range_current
                    height_ref = height_ref_current
                    first_match = False
                
                assert (np.all(x_range == x_range_current) and
                        np.all(y_range == y_range_current) and
                        height_ref == height_ref_current)
                    
                if match[2] == 'Weibull-A':
                    Weibull_A_grds.append((i_sector, height_ref, data))
                
                if match[2] == 'Weibull-k':
                    Weibull_k_grds.append((i_sector, height_ref, data))  
                    
                if match[2] == 'Sector frequency':
                    frequency_grds.append((i_sector, height_ref, data))
                    
        num_sector = len(Weibull_A_grds)
        num_x = len(x_range)
        num_y = len(y_range)
        
        Weibull_A = np.zeros((num_x, num_y, num_sector))
        for grd_info in Weibull_A_grds:
            Weibull_A[:, :, grd_info[0]] = grd_info[2]
        
        Weibull_k = np.zeros((num_x, num_y, num_sector))
        for grd_info in Weibull_k_grds:
            Weibull_k[:, :, grd_info[0]] = grd_info[2]
        
        
        frequency = np.zeros((num_x, num_y, num_sector))
        for grd_info in frequency_grds:
            frequency[:, :, grd_info[0]] = grd_info[2]
          
        wd_binned = np.linspace(0, 360-360./num_sector, num_sector)
        
        return SiteCondition(Weibull_A, Weibull_k, frequency, 
                             wd_binned, height_ref, z0, x_range, y_range)
        
                    
if __name__ == '__main__':
    Akf_file = '../../inputs/HornsRevAkf.txt'
    Akf = np.loadtxt(Akf_file)
    height_ref = 62
    
    site_condition = SiteCondition(Akf[:, 1], Akf[:, 2], Akf[:, 3], Akf[:, 0],
                                   height_ref=height_ref)
    print(site_condition.getAkf())
         
    grd_path = '../../inputs/Middelgrunden/'
    site_condition_MG = SiteCondition.from_WAsP_grd(grd_path)
    
    num_points = 1000
    x = site_condition_MG.x_range[0] + np.random.random(num_points)*(
            site_condition_MG.x_range[-1] - site_condition_MG.x_range[0])
    
    y = site_condition_MG.y_range[0] + np.random.random(num_points)*(
            site_condition_MG.y_range[-1] - site_condition_MG.y_range[0])
    
    wd = np.random.random(num_points)*3600
    
    A, k, f, height_ref = site_condition_MG.getAkf(x, y, wd)
    
    