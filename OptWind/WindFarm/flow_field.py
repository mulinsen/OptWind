# -*- coding: utf-8 -*-
import numpy as np

class FlowField(object):
    def __init__(self, wf_design, getAkf, height_ref, alpha=0.04,
                 ws_binned=np.linspace(1, 30, 30),
                 wd_binned=np.linspace(0, 330, 12),
                 z0=0.01
                 ):
        
        self.complete_layout = wf_design.complete_layout
        self.getAkf  = getAkf
        self.height_ref = height_ref
        self.alpha = alpha
        self.ws_binned = ws_binned
        self.wd_binned = wd_binned
        self.z0=z0
        self.wt_types = wf_design.wt_types
        self.wind_shear_multi = self.wind_shear_log(self.complete_layout[:, 2], 
                                                    height_ref)
        
        self.num_wt = wf_design.num_wt
        self.num_ws_bin = len(ws_binned)
        self.num_wd_bin = len(wd_binned)
        self.ws_bin_size = ws_binned[1] - ws_binned[0]
        self.wd_bin_size = wd_binned[1] - wd_binned[0]
        self.R_list = wf_design.D_list/2
        self.Ar_list = wf_design.Ar_list
        self.type_list = np.array([int(t) for t in self.complete_layout[:, 3]], 
                                   dtype='int')
        
        self.M_ijl = np.zeros((self.num_wt, self.num_wt, self.num_wd_bin))
        self.v_ikl_ideal = np.zeros(
                (self.num_wt, self.num_ws_bin, self.num_wd_bin))
        self.v_ikl_real = np.zeros(
                (self.num_wt, self.num_ws_bin, self.num_wd_bin))
        self.pdf_ikl = np.zeros(
                (self.num_wt, self.num_ws_bin, self.num_wd_bin))
        
    def wind_shear_log(self, H, H_ref):
        return np.log(H/self.z0)/np.log(H_ref/self.z0)
    
    def change_layout(self, complete_layout_new):
        """ Assume only locations of turbines changed, and number, hub-height
        and types of turbines remained the same."""
        self.complete_layout = complete_layout_new
  
    def cal_flow_field(self):
        ######################################################################
        # 1. calculate ideal wind speed
        v_ik = np.expand_dims(
                np.matmul(np.expand_dims(self.wind_shear_multi, axis=-1),
                         np.expand_dims(self.ws_binned, axis=0)), axis=-1)
        
        self.v_ikl_ideal = np.concatenate([v_ik
                                           for l_wd in range(self.num_wd_bin)],
                                          axis=-1)
        
        ######################################################################
        # 2. calculate pdf of local ideal wind speed
        x_il = np.concatenate([np.expand_dims(self.complete_layout[:, 0], axis=-1)
                               for l_wd in range(self.num_wd_bin)], axis=-1)
        y_il = np.concatenate([np.expand_dims(self.complete_layout[:, 1], axis=-1)
                               for l_wd in range(self.num_wd_bin)], axis=-1)
        wd_il = np.concatenate([np.expand_dims(self.wd_binned, axis=0)
                               for l_wt in range(self.num_wt)], axis=0)

        A_il, k_il, f_il = self.getAkf(x_il, y_il, wd_il)
        
        for k_ws in range(self.num_ws_bin):
            self.pdf_ikl[:, k_ws, :] = self.cal_pdf_Weibull(
                    self.ws_binned[k_ws]*np.ones_like(A_il), A_il, k_il) * f_il
            
            
        dist_down = np.zeros((self.num_wt, self.num_wt, self.num_wd_bin))
        dist_cross = np.zeros((self.num_wt, self.num_wt, self.num_wd_bin))
        R_wake = np.zeros((self.num_wt, self.num_wt, self.num_wd_bin))
        A_ol = np.zeros((self.num_wt, self.num_wt, self.num_wd_bin))
        #######################################################################
        # 3. calculate real wind speed
        # calculate M_ijl matrix
        for l_wd in range(self.num_wd_bin):
            rotate_angle = (270 - self.wd_binned[l_wd])*np.pi/180.0
            cos_rotate = np.cos(rotate_angle)
            sin_rotate = np.sin(rotate_angle)
        
            x_rotated = (self.complete_layout[:, 0]*cos_rotate + 
                         self.complete_layout[:, 1]*sin_rotate)
            y_rotated = (self.complete_layout[:, 1]*cos_rotate - 
                         self.complete_layout[:, 0]*sin_rotate)
            
            downwind_order = np.argsort(x_rotated)
            
            for i_up in range(self.num_wt-1):
                index_up = downwind_order[i_up]
                index_down = downwind_order[i_up+1:]
                
                dist_down[index_down, index_up, l_wd] = (
                        x_rotated[index_down] - x_rotated[index_up])
                
                dist_cross[index_down, index_up, l_wd] = np.sqrt(
                        (y_rotated[index_down] - y_rotated[index_up])**2 +
                        (self.complete_layout[index_down, 2] - 
                         self.complete_layout[index_up, 2])**2)
                
                R_wake[index_down, index_up, l_wd] = (
                        self.alpha*dist_down[index_down, index_up, l_wd] + 
                        self.R_list[index_up])
        R = np.concatenate([
                np.expand_dims(self.R_list[i_wt]*np.ones(
                        (self.num_wt, self.num_wd_bin)), axis=1) 
                for i_wt in range(self.num_wt)], axis=1) 
        R1 = np.concatenate([
                np.expand_dims(self.R_list[i_wt]*np.ones(
                        (self.num_wt, self.num_wd_bin)), axis=0) 
                for i_wt in range(self.num_wt)], axis=0)
        A_ol = self.cal_overlapping_area(R_wake, 
                                         R,
                                         dist_cross)
        
        self.M_ijl = np.where(dist_down>0, 
                (A_ol/(np.pi*R**2))**2 / (1 + self.alpha*dist_down/R1)**4,
                 0)
                
        # calculate N_jk matrix
        v_jk = self.v_ikl_ideal[:, :, 0]
        N_jk = np.zeros_like(v_jk)
        for m_type in set(self.type_list):
            index_cal = self.type_list == m_type
            N_jk[index_cal, :] = self.wt_types[m_type].get_Ct(
                    v_jk[index_cal, :])
        N_jk = v_jk**2*(1-np.sqrt(1-N_jk))**2
        
        self.v_ikl_real = self.v_ikl_ideal - np.sqrt(
                np.einsum('jk, ijl->ikl', N_jk, self.M_ijl))  
        
        
    def cal_flow_field_naive(self):
        ######################################################################
        # 1. calculate ideal wind speed
        v_ik = np.expand_dims(
                np.matmul(np.expand_dims(self.wind_shear_multi, axis=-1),
                         np.expand_dims(self.ws_binned, axis=0)), axis=-1)
        
        self.v_ikl_ideal = np.concatenate([v_ik
                                           for l_wd in range(self.num_wd_bin)],
                                          axis=-1)
        
        ######################################################################
        # 2. calculate pdf of local ideal wind speed
        x_il = np.concatenate([np.expand_dims(self.complete_layout[:, 0], axis=-1)
                               for l_wd in range(self.num_wd_bin)], axis=-1)
        y_il = np.concatenate([np.expand_dims(self.complete_layout[:, 1], axis=-1)
                               for l_wd in range(self.num_wd_bin)], axis=-1)
        wd_il = np.concatenate([np.expand_dims(self.wd_binned, axis=0)
                               for l_wt in range(self.num_wt)], axis=0)

        A_il, k_il, f_il = self.getAkf(x_il, y_il, wd_il)
        
        for k_ws in range(self.num_ws_bin):
            self.pdf_ikl[:, k_ws, :] = self.cal_pdf_Weibull(
                    self.ws_binned[k_ws]*np.ones_like(A_il), A_il, k_il) * f_il
            
            
    
        #######################################################################
        # 3. calculate real wind speed
        # calculate M_ijl matrix
        for l_wd in range(self.num_wd_bin):
            rotate_angle = (270 - self.wd_binned[l_wd])*np.pi/180.0
            cos_rotate = np.cos(rotate_angle)
            sin_rotate = np.sin(rotate_angle)
        
            x_rotated = (self.complete_layout[:, 0]*cos_rotate + 
                         self.complete_layout[:, 1]*sin_rotate)
            y_rotated = (self.complete_layout[:, 1]*cos_rotate - 
                         self.complete_layout[:, 0]*sin_rotate)
            
            downwind_order = np.argsort(x_rotated)
            
            for i_up in range(self.num_wt-1):
                index_up = downwind_order[i_up]
                index_down = downwind_order[i_up+1:]
                
                dist_down = x_rotated[index_down] - x_rotated[index_up]
                dist_cross = np.sqrt(
                        (y_rotated[index_down] - y_rotated[index_up])**2 +
                        (self.complete_layout[index_down, 2] - 
                         self.complete_layout[index_up, 2])**2)
                
                R_wake = self.alpha*dist_down + self.R_list[index_up]
                
                A_ol = self.cal_overlapping_area(R_wake, 
                                                 self.R_list[index_down],
                                                 dist_cross)
                self.M_ijl[index_down, index_up, l_wd] = (
                        (A_ol/self.Ar_list[index_down])**2 /
                        (1 + self.alpha*dist_down/self.R_list[index_up])**4)
                
        # calculate N_jk matrix
        v_jk = self.v_ikl_ideal[:, :, 0]
        N_jk = np.zeros_like(v_jk)
        for m_type in set(self.type_list):
            index_cal = self.type_list == m_type
            N_jk[index_cal, :] = self.wt_types[m_type].get_Ct(
                    v_jk[index_cal, :])
        N_jk = v_jk**2*(1-np.sqrt(1-N_jk))**2
        
        self.v_ikl_real = self.v_ikl_ideal - np.sqrt(
                np.einsum('jk, ijl->ikl', N_jk, self.M_ijl))
  
          

    def cal_overlapping_area(self, R1, R2, d):
        """ Calculate the overlapping area of two circles with radius R1 and
        R2, centers distanced d.

        The calculation formula can be found in Eq. (A1) of :
        [Ref] Feng J, Shen WZ, Solving the wind farm layout optimization
        problem using Random search algorithm, Reneable Energy 78 (2015)
        182-192
        Note that however there are typos in Equation (A1), '2' before alpha
        and beta should be 1.

        Parameters
        ----------
        R1: array:float
            Radius of the first circle [m]

        R2: array:float
            Radius of the second circle [m]

        d: array:float
            Distance between two centers [m]

        Returns
        -------
        A_ol: array:float
            Overlapping area [m^2]
        """
        # treat all input as array
        R1, R2, d = np.array(R1), np.array(R2), np.array(d),
        A_ol = np.zeros_like(R1)
        p = (R1 + R2 + d)/2.0

        # make sure R_big >= R_small
        Rmax = np.where(R1 < R2, R2, R1)
        Rmin = np.where(R1 < R2, R1, R2)

        # full wake cases
        index_fullwake = (d<= (Rmax -Rmin))
        A_ol[index_fullwake] = np.pi*Rmin[index_fullwake]**2

        # partial wake cases
        index_partialwake = np.logical_and(d > (Rmax -Rmin),
                                           d < (Rmin + Rmax))

        alpha = np.arccos(
           (Rmax[index_partialwake]**2.0 + d[index_partialwake]**2
            - Rmin[index_partialwake]**2)
            /(2.0*Rmax[index_partialwake]*d[index_partialwake]) )

        beta = np.arccos(
           (Rmin[index_partialwake]**2.0 + d[index_partialwake]**2
            - Rmax[index_partialwake]**2)
            /(2.0*Rmin[index_partialwake]*d[index_partialwake]) )

        A_triangle = np.sqrt( p[index_partialwake]*
                             (p[index_partialwake]-Rmin[index_partialwake])*
                             (p[index_partialwake]-Rmax[index_partialwake])*
                             (p[index_partialwake]-d[index_partialwake]) )

        A_ol[index_partialwake] = (  alpha*Rmax[index_partialwake]**2
                                   + beta*Rmin[index_partialwake]**2
                                   - 2.0*A_triangle )

        return A_ol
    
    def cal_pdf_Weibull(self, v, A, k):
        return ((k / A) * (v / A) ** (k - 1) * np.exp(-(v / A) ** k))
            
        