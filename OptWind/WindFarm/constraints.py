# -*- coding: utf-8 -*-
import numpy as np

class Constraints(object):
    def __init__(self, x_min, x_max, y_min, y_max,
                       inclusive_boundaries = [],
                       exclusive_boundaries = [], 
                       minimal_distance= None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.inclusive_boundaries = inclusive_boundaries
        self.exclusive_boundaries = exclusive_boundaries
        self.minimal_distance = minimal_distance
        
    def check_constraints_bool(self, complete_layout):
        xy_array = complete_layout[:, :2]
        
        return (self.check_bounds_bool(xy_array) and 
                self.check_minimal_distance_bool(xy_array))
    
    def check_constraints_float(self, complete_layout):
        xy_array = complete_layout[:, :2]
        
        bounds_flag, bounds_viol = self.check_bounds_float(xy_array)
        mini_dist_flag, mini_dist_viol = self.check_minimal_distance_float(
                xy_array)
        
        return (bounds_flag and mini_dist_flag,
                bounds_viol + mini_dist_viol)
        
    
    def check_bounds_bool(self, xy_array):
        if np.any(xy_array[:, 0] < self.x_min):
            return False
        if np.any(xy_array[:, 0] > self.x_max):
            return False
        if np.any(xy_array[:, 1] < self.y_min):
            return False
        if np.any(xy_array[:, 1] > self.y_max):
            return False
        return True
    
    def check_bounds_float(self, xy_array):
        violation_degree = 0.0
        
        violation_degree += np.sum(np.where(xy_array[:, 0] < self.x_min,
                                            self.x_min - xy_array[:, 0], 
                                            0.0))
        violation_degree += np.sum(np.where(xy_array[:, 0] > self.x_max,
                                            xy_array[:, 0] - self.x_max, 
                                            0.0))
        violation_degree += np.sum(np.where(xy_array[:, 1] < self.y_min,
                                            self.y_min - xy_array[:, 1], 
                                            0.0))
        violation_degree += np.sum(np.where(xy_array[:, 1] > self.y_max,
                                            xy_array[:, 1] - self.y_max, 
                                            0.0))
        return (violation_degree>0, violation_degree)
        
    def check_minimal_distance_bool(self, xy_array):
        if self.minimal_distance is None:
            return True
        
        num_wt = xy_array.shape[0]
        
        for i_wt in range(num_wt - 1):
            dist = np.sqrt((xy_array[i_wt, 0] - xy_array[i_wt+1:, 0])**2 +
                           (xy_array[i_wt, 1] - xy_array[i_wt+1:, 1])**2)
            if np.any(dist > self.minimal_distance):
                return False
        
        return True
    
    def check_minimal_distance_float(self, xy_array):
        if self.minimal_distance is None:
            return (True, 0.0)
        
        num_wt = xy_array.shape[0]
        
        violation_degree = 0.0
        
        for i_wt in range(num_wt - 1):
            dist = np.sqrt((xy_array[i_wt, 0] - xy_array[i_wt+1:, 0])**2 +
                           (xy_array[i_wt, 1] - xy_array[i_wt+1:, 1])**2)
            
            violation_degree += np.sum(np.where(dist < self.minmal_distance,
                                                self.minmal_distance - dist,
                                                0.0))   
        
        return (violation_degree > 0.0, violation_degree/num_wt)
            
        
        
        