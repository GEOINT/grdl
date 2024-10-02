# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:16:54 2024

@author: User
"""
import numpy as np
class phd_return():
    '''
    This class is meant to parse the meta data of SICD and return the valid regions
    of the data. You would want to use this for the purpose of making the subapertures
    of the psuedo phase history data from a SICD
    '''
    def __init__(self, meta):
        self.meta = meta
        self.get_oversample()
        
    def get_oversample(self):
        az_oversample = 1/( self.meta.Grid.Col.ImpRespBW*self.meta.Grid.Col.SS)
        rg_oversample = 1/( self.meta.Grid.Row.ImpRespBW*self.meta.Grid.Row.SS)
        #az_oversample = 1
        #rg_oversample = 1
        self.rg_os_pixel_frac = 1 - 1/rg_oversample
        self.az_os_pixel_frac = 1 - 1/az_oversample
        a = 1
        
    def get_phd_pixels(self, chip_shape):
        nrows = chip_shape[0]
        ncols = chip_shape[1]
        
        rstart = np.round( nrows*self.rg_os_pixel_frac/2 ).astype(int)
        cstart = np.round( ncols*self.az_os_pixel_frac/2 ).astype(int)
        
        return rstart, cstart
        
        
        
        
        