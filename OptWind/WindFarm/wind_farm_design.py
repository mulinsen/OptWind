# -*- coding: utf-8 -*-
import numpy as np

class WindFarmDesign(object):
    def __init__(self, complete_layout, wt_types, name='unknown'):
        self.complete_layout = complete_layout
        self.num_wt = complete_layout.shape[0]
        self.wt_types = wt_types
        self.num_wt_type = len(wt_types)
        self.D_list = np.array([wt_types[int(t)].get_D()
                                for t in self.complete_layout[:, 3]])
        self.Ar_list = np.pi*(self.D_list/2)**2
    
    
        
        