# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.ElementTree as ET

class WindTurbine(object):
    """ Wind turbine.
    
    Parameters
    ----------
    name : string
        Name of this type of WT.
        
    ID : string
        Unique ID for this turbine type.
    
    diameter : float
        Rotor diameter [m].
        
    hub_height : float
        Default hub height [m].
    
    rated_power : float
        Rated power [MW].
        
    num_operation_modes : integer
        number of operation modes, default = 1.
        
    ws_cutins : float list (len = num_operation_modes)
        Cut-in wind speeds [m/s]. ws_cutins[i] is the cutin wind speed for the
        ith operation modes.
    
    ws_cutouts : float list (len = num_operation_modes)
        cut-out wind speed [m/s]. ws_cutins[i] is the cutout wind speed for the
        ith operation modes.
    
    char_data_tables : list of float arrays with shape = [n_points, 3] (len = 
        num_operation_modes)
        Data table representing the characteristics with n_points data points.
        char_data_table[:, 0] : wind speed range [m/s];
        char_data_table[:, 1] : Ct (thrust coefficient) curve [-];
        char_data_table[:, 2] : power curve [MW].
    
    air_densitys : float list (len = num_operation_modes), optional 
        (default = 1.225)
        Air densitys corresponding to the data tables [kg/m3].
         
    rotor_area : float, derived
        Rotor area [m2].
        
    Methods:
    --------
    get_D(): get the rotor diameter [m]
    get_ws_cutin: get the cut-in wind speed [m/s]
    get_ws_cutout: get the cut-out wind speed [m/s]
    
    get_power(ws): power output for inflow wind speed - ws [m/s].
    
    get_Ct(ws): thrust coefficient for inflow wind speed - ws [m/s].
    
    pltPowerCtCurve(): plot the power and Ct curves in one figure and saved.
    
    Examples:
    ---------
    >>> from wind_turbine import WindTurbine
    >>> dt = np.array([[4, 0.8, 0.0], [10, 0.8, 2.0],  [25, 0.8, 2.0]])
    >>> WT = WindTurbine('ideal', None, 80.0, 70.0, 2., [4.], [25.], [dt])
    >>> print(WT.name)
    ideal
    >>> print(WT.rated_power)
    2.0
    >>> print(np.round(WT.rotor_area, decimals=4))
    5026.5482
    >>> WT.get_power(WT.get_ws_cutin())
    0.0
    >>> WT.get_power(WT.get_ws_cutout())
    2.0
    >>> WT.get_power(7.0)
    1.0
    >>> print(np.round(WT.get_Ct(7.350), decimals=1))
    0.8
    >>> WT.get_Ct(25.01)
    0.0
     
    """
    
    def __init__(self, name, ID, diameter, hub_height, rated_power, 
                 ws_cutins, ws_cutouts, char_data_tables,
                 num_operation_modes=1,
                 air_densities=[1.225],
                 mode_names=None): 
        self.name = name
        self.ID = ID
        self.diameter = diameter
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.ws_cutins = ws_cutins
        self.ws_cutouts = ws_cutouts
        self.char_data_tables = char_data_tables
        self.num_operation_modes = num_operation_modes
        self.air_densities = air_densities
        self.mode_names = mode_names
        self.rotor_area = np.pi*(diameter/2.0)**2	    
        
        
    def __str__(self):
        WT_string = ('[Wind turbine - an object of WT_cls]\n' 
                     + 'Name: ' + self.name + '; \n' 
                     + 'ID: ' + self.ID + '; \n' 
                     + 'Diameter: ' + str(self.diameter) + ' [m];\n' 
                     + 'Rated power: ' + str(self.rated_power) + ' [MW];\n' 
                     + 'Default hub height: ' + str(self.hub_height) + ' [m];\n'
                     + 'Cut-in wind speed: ' + str(self.ws_cutins[0]) + ' [m/s];\n'
                     + 'Cut-out wind speed: ' + str(self.ws_cutouts[0]) + ' [m/s];\n'
                     + 'V [m/s]\tCt [-]\tPower [MW]:\n'
                     + str(self.char_data_tables[0])
                     )
        return WT_string
   
    
    
    @classmethod
    def from_WAsP_element(cls, WT_element):
        """ Parse the WT element inside the inventory.xml of .wwh file to 
        initilize an WT_cls object.
        
        Parameters
        ----------
        WT_element : ET element
            Extracted from .xml file in the .wwh file from WindPRO.
        
        Returns
        -------
        WT : an object of WT_cls.
        """
        name = WT_element.attrib['Description']
        ID = WT_element.attrib['ID']
        
        wtg = WT_element.find('.//WindTurbineGenerator')
        diameter = np.float(wtg.attrib['RotorDiameter'])
        hub_height = np.float(wtg.find('SuggestedHeights').find('Height').text)
        turbine_modes = wtg.findall('PerformanceTable')
        num_modes = len(turbine_modes)
        ws_cutins = [0.0 for i in range(num_modes)]
        ws_cutouts = [0.0 for i in range(num_modes)]
        air_densities = [1.225 for i in range(num_modes)]
        dts = [None for i in range(num_modes)]
        
        for i in range(num_modes):
            turbine_mode = turbine_modes[i]
            air_densities[i] = np.float(turbine_mode.attrib['AirDensity'])
            
            ws_cutins[i] = np.float(turbine_mode.find(
                    'StartStopStrategy').attrib['LowSpeedCutIn'])
            
            ws_cutins[i] = np.float(turbine_mode.find(
                    'StartStopStrategy').attrib['HighSpeedCutOut'])
            
            data_points = turbine_mode.findall('.//DataPoint')
            
            dt = [[np.float(dp.attrib['WindSpeed']),
                   np.float(dp.attrib['ThrustCoEfficient']),
                   np.float(dp.attrib['PowerOutput'])]
                   for dp in data_points]
    
            dts[i] = np.array(dt)
            
            if np.max(dts[i][:, 2]) > 10^5:
                dts[i][:, 2] = dts[i][:, 2]/1.E6    # from w to MW
            elif np.max(dts[i][:, 2]) > 100:
                dts[i][:, 2] = dts[i][:, 2]/1.E3   # from kw to MW
            
        rated_power = np.max(dts[0][:, 2])
        
        WT = cls(name, ID, diameter, hub_height, rated_power, 
                 ws_cutins, ws_cutouts, dts,
                 num_modes,
                 air_densities)
        
        return WT
    
    @classmethod
    def from_WAsP_wtg(cls, wtg_file):
        """ Parse the .wtg file (xml) to initilize an WT_cls object.
        
        Parameters
        ----------
        wtg_file : string
            A string denoting the .wtg file, which is exported from WAsP.
        
        Returns
        -------
        WT : an object of WT_cls
        """
        tree = ET.parse(wtg_file)
        root = tree.getroot()
        # Reading data from wtg_file
        name = root.attrib['Description']
        diameter = np.float(root.attrib['RotorDiameter'])
        hub_height = np.float(root.find('SuggestedHeights').find('Height').text)
        ws_cutin = np.float(root.find('PerformanceTable'
                   ).find('StartStopStrategy').attrib['LowSpeedCutIn'])  
        ws_cutout = np.float(root.find('PerformanceTable'
                    ).find('StartStopStrategy').attrib['HighSpeedCutOut'] )
        
        i_point = 0
        for DataPoint in root.iter('DataPoint'):
            i_point = i_point + 1
            ws = np.float(DataPoint.attrib['WindSpeed'])
            Ct = np.float(DataPoint.attrib['ThrustCoEfficient'])
            power = np.float(DataPoint.attrib['PowerOutput'])/1.E6 # from W to MW
            if i_point == 1:
                dt = np.array([[ws, Ct, power]])
            else:
                dt = np.append(dt, np.array([[ws, Ct, power]]), axis=0)
    
        rated_power = np.max(dt[:, 2])
            
        WT = cls(name, None, diameter, hub_height, rated_power, 
                 [ws_cutin], [ws_cutout], [dt])
        
        return WT
    
    @classmethod
    def from_WindPRO_element(cls, WT_element):
        """ Parse the ET element from WindPRO to initilize an WT_cls object.
        
        Parameters
        ----------
        WT_element : ET element
            Extracted from .xml file in the .optireq file from WindPRO.
        
        Returns
        -------
        WT : an object of WT_cls
        """
        name = WT_element.attrib['TurbineTypeUID']
        diameter = np.float(WT_element.attrib['RotorDiameter'])
        hub_height = np.float(WT_element.attrib['HubHeight'])
        
        turbine_modes = WT_element.findall('TurbineMode')
        num_modes = len(turbine_modes)
        ws_cutins = [0.0 for i in range(num_modes)]
        ws_cutouts = [0.0 for i in range(num_modes)]
        air_densities = [1.225 for i in range(num_modes)]
        dts = [None for i in range(num_modes)]
        mode_ids = [None for i in range(num_modes)]
        for i in range(num_modes):
            turbine_mode = turbine_modes[i]
            air_densities[i] = np.float(turbine_mode.attrib['AirDensity'])
            mode_ids[i] = turbine_mode.attrib['ModeID']
            ws_cutouts[i] = turbine_mode.attrib['StopWindSpeed']
            
            power_curve = turbine_mode.find('PowerCurve')
            thrust_curve = turbine_mode.find('ThrustCurve')
            
            pc = []
            for data in power_curve.findall('Data'):
                ws = np.float(data.attrib['windSpeed'])
                power = np.float(data.attrib['power'])
                pc.append([ws, power])
                
            tc = []
            for data in thrust_curve.findall('Data'):
                ws = np.float(data.attrib['windSpeed'])
                Ct = np.float(data.attrib['CT'])
                tc.append([ws, Ct])
                
            if pc[0][1] == 0:
                pc = pc[1:]
            
            assert len(pc) == len(tc)
            pc = np.array(pc)
            tc = np.array(tc)
            
            dt = np.hstack((tc, pc))
            dt = dt[:, (0, 1, 3)]
            
            dts[i] = dt
            ws_cutins[i] = tc[0, 0]
    
        rated_power = np.max(dts[0][:, 2])
        
        # change kW to MW for wind speed
        if rated_power > 100:
            rated_power = rated_power/1000
            for dt in dts:
                dt[:, -1] = dt[:, -1]/1000
            
        WT = cls(name, None, diameter, hub_height, rated_power, 
                 ws_cutins, ws_cutouts, dts,
                 num_modes,
                 air_densities,
                 mode_ids)
        
        return WT
        
   
    def get_D(self):
        """ Get the rotor diameter [m].
        """
        return self.diameter
    
    
    def get_H(self):
        """ Get the default hub height [m].
        """
        return self.hub_height
    
    
    def get_rotor_area(self):
        """ Get the rotor area [m2].
        """
        return self.rotor_area
   
          
    def get_ws_cutin(self, mode=0):
        """ Get the cut-in wind speed [m/s].
        """
        return self.ws_cutins[mode]
   
     
    def get_ws_cutout(self, mode=0):
        """ Get the cut-out wind speed [m/s].
        """
        return self.ws_cutouts[mode]     
                        
        
    def get_power(self, ws, mode=0):
        """ Get power output when the inflow wind speed is ws.
        
        Parameters
        ----------
        ws : (multi-dimensional) array 
            Inflow wind speed [m/s].
           
        mode : integer
            Operation mode (default = 0)
            
        Returns
        -------
        power : (multi-dimensional) array, same shape as ws - 
            Power output [MW].
        """
        power = np.interp(ws, self.char_data_tables[mode][:, 0], 
                          self.char_data_tables[mode][:, 2], 
                          left=0.0, right=0.0)
        return power
        
        
    def get_Ct(self, ws, mode=0):
        """ Get Ct when the inflow wind speed is ws.
        
        Parameters
        ----------
        ws : (multi-dimensional) array 
            Inflow wind speed [m/s].
            
        mode : integer
            Operation mode (default = 0)
        
        Returns
        -------
        Ct : (multi-dimensional) array, same shape as ws 
            Ct [-].
        """
        Ct = np.interp(ws, self.char_data_tables[mode][:, 0], 
                       self.char_data_tables[mode][:, 1], 
                       left=0.0, right=0.0)
        # do not allow Ct to be greater than 1
        Ct =  np.where(Ct>1, 1, Ct)
        Ct = Ct*1
        return Ct
        
    
    def pltPowerCtCurve(self, SavePath=None, mode=0):        
        """ Plot power curve and Ct curve of this turbine.
        
        And save the figure named as the turbine's name (self.name) in the 
        folder given by SavePath or the current working directory (default).
        """
        import matplotlib.pyplot as plt
        
        if SavePath is None:
            SavePath = './' # default: the current working directory
        
        fig, ax1 = plt.subplots()
        ax1.plot(self.char_data_tables[mode][:, 0], 
                 self.char_data_tables[mode][:, 2], 'b-')
        ax1.set_xlabel('Wind speed [m/s]')
        ax1.set_ylabel('Power [MW]', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        plt.ylim(0, self.rated_power*1.1)
        
        ax2 = ax1.twinx()
        ax2.plot(self.char_data_tables[mode][:, 0], 
                 self.char_data_tables[mode][:, 1], 'r-')
        ax2.set_ylabel('Ct [-]', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        plt.ylim(0, 1)
        plt.title('Wind turbine - ' + self.name)
        #plt.show()
        fig.set_size_inches(5, 5)
        fig.set_dpi(100)
        fig.savefig(SavePath + 'Wind turbine - ' +self.name  
                    + ' _mode '+ str(mode) + '.tiff')

        
if __name__ == '__main__':
    import doctest

    
    doctest.testmod() 
    
    wtg_file = '../../inputs/Middelgrunden/Bonus 2 MW.wtg'
    WT1 = WindTurbine.from_WAsP_wtg(wtg_file)
    WT1.pltPowerCtCurve(SavePath='../../outputs/Middelgrunden/')
   
    
