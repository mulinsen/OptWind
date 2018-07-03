# -*- coding: utf-8 -*-
import numpy as np

class AEP(object):  
    def cal_AEP(flow_field, availability = 1.0, num_hrs_a_year = 8760):
        """ Calculate gross AEP, net AEP of the wind farm in flow_field.

        Parameters
        ----------
        flow_field: :any:`FlowField`
            A object hosts the site_condition and wind_farm, and the flow field
            information calcualted by considering wake effect. 

        Returns
        -------
        AEP_gross: array:float
            Gross AEP values of each turbine in the wind farm.

        AEP_net: array:float
            Net AEP values of each turbine in the wind farm.
        """
        
        power_gross = np.zeros_like(flow_field.v_ikl_ideal)
        power_net = np.zeros_like(flow_field.v_ikl_ideal)
        
        # Calculate gross and net power
        for j_type in set(flow_field.type_list):
            match_index = flow_field.type_list == j_type
            
            power_gross[match_index, :, :] = (
                    flow_field.wt_types[j_type].get_power(
                            flow_field.v_ikl_ideal[match_index, :, :]))
                    
            power_net[match_index, :, :] = (
                    flow_field.wt_types[j_type].get_power(
                            flow_field.v_ikl_real[match_index, :, :]))
        
        
        
        AEP_gross_il = ( np.sum(power_gross * flow_field.pdf_ikl *
                                     flow_field.ws_bin_size *
                                     flow_field.wd_bin_size, axis=1) *
                         num_hrs_a_year * availability)
         
        AEP_net_il = ( np.sum(power_net * flow_field.pdf_ikl *
                                     flow_field.ws_bin_size *
                                     flow_field.wd_bin_size, axis= 1) *
                         num_hrs_a_year * availability)
        
        AEP_gross_i = np.sum(AEP_gross_il, axis=1)
        AEP_net_i = np.sum(AEP_net_il, axis=1)

        AEP_gross = np.sum(AEP_gross_i)
        AEP_net = np.sum(AEP_net_i)
        
        return (AEP_gross, AEP_net)