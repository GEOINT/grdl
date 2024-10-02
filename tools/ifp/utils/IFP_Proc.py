# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:10:11 2024

@author: User
"""

import matplotlib
matplotlib.rcParams['figure.dpi'] = 200 

import json

from utils.metasetup import Meta_Setup

from utils.getcoordinates import Get_Coordinates

from utils.getpolargrid import Get_Polar_Grid

from utils.pfa import PFA
#from utils.rma_algo import RMA_Algo
from utils.compress_phd_data import compress_phd_data

from utils.make_sicd import Make_SICD


class IFP_Proc():
    '''
    This is the runtime class to run the image formation process for synthetic aperture
    radar data. 
    
    data inputs: NGA standard V 1.1.0 compliant Compensated Phase History Data (CPHD)
    outputs: NGA standard V 1.3.0 compliant Sensor Independent Complex Data (SICD)
    '''
    
    def __init__(self, proc_inputs ):
        '''
        proc_inputs: DICT
            dictionary input of the processing information.
        '''
        
        ### fill the necessary inputs
        self.file_name = proc_inputs['file_name']
        self.data_path = proc_inputs['data_path']
        ### the amount of range compression, this determines the image size
        self.range_compression = proc_inputs['range_compression']
        ### the amount of azimuth compression, this determines the image size
        self.azimuth_compression = proc_inputs['azimuth_compression']
        ### do we inscribe a grid on the polar data or circumscribe
        self.Grid_Inscription = eval(proc_inputs['Grid_Inscription'])
        ### the Image formation plane, usually slant or ground
        self.IFP_Projection = proc_inputs['IFP_Projection']
        
    def populate_metadata(self):
        '''
        
        Populate the metadata. Load the CPHD and fill all the per vector paramters 
        for each pulse. this provides the position, time and input vectors for the 
        antenna phase center (APC)

        Returns
        -------
        None.

        '''
        self.metadata = Meta_Setup( self.data_path + self.file_name )
        
    def populate_geometry(self, plot_collection_geo = False):
        
        slant = True
        if self.IFP_Projection != 'Slant':
            slant = False
        
        self.geo = Get_Coordinates(self.metadata)
        self.geo.get_coordinates(slant = slant)
        if plot_collection_geo == True:
            self.geo.plot_collection_geometry()
        
    def populate_polargrid(self):
        
        self.polar_grid = Get_Polar_Grid(self.metadata,
                                         self.geo, 
                                         self.range_compression, 
                                         self.azimuth_compression, 
                                         self.Grid_Inscription )
        self.polar_grid.polar_grid()
        self.polar_grid.plot_polar_grid()
        a = 1
        
    def perform_pfa(self):

        pfa = PFA( self.metadata, self.geo, self.polar_grid )
        print('begin interpolation')
        phd_pfa = pfa.perform_pfa(self.metadata.reader)
        
        self.sicd = compress_phd_data(phd_pfa)
        
    def perform_rma(self):
        RMA_Algo(self.metadata, self.geo )
        
    def plot_image(self):
        import matplotlib.pyplot as plt
        from sarpy.visualization.remap import Density
        
        rm = Density(bit_depth = 16)
        
        plt.figure()
        plt.imshow(rm(self.sicd), aspect='auto', cmap='gray')
        plt.title(f'Processed image, {self.file_name}')
        
    def save_sicd(self):
        sicd_build = Make_SICD(self.metadata, self.geo, self.polar_grid)
        sicd_build.populate_fields()
        sicd_out_name = self.file_name.strip('.cphd') + '_smalley_SICD.nitf'
        output_name = 'D:/SAR_Data/ifpSmallee/' + sicd_out_name
        sicd_build.write_sicd( self.sicd, output_name )