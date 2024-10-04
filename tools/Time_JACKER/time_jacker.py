# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:55:38 2024

@author: User
"""

import cv2
import numpy as np

from grdl.SAR.processing.sub_aperture import sub_aperture
from grdl.image_processing.std_filter import std_filter
from grdl.image_segmentation.binary_segs import Binary_Segs
from grdl.SAR.plotting.plot_sar_image import plot_sar_image

class Time_JACKER():
    '''
    Time_JACKER, this is an approach to extract time based / aperture based glints
    based on the change in magnitude between apertures 
    
    WHO'S READY FOR THE TIME JACKER
    '''
    def __init__(self, num_aps=4, debug = False):
        
        self.num_aps = num_aps
        self.debug = debug

    
    def tj(self, data, meta, thresh=50):
        '''
        In order to JACK time one must pass in the data, complex sicd format, and the
        meta data from the SARPY
        '''
        
        sa = sub_aperture(meta)
        time_sa = sa.get_time_sub_aperture(data, num_aps = self.num_aps)
        
        ### call the bs
        bs = Binary_Segs()
        
        self.time_deviation = np.zeros( data.shape )
        
        for i in range( self.num_aps-1):
            norm_i = (time_sa[i]-np.mean( time_sa[i]))/np.std(time_sa[i])
            norm_i1 = (time_sa[i+1]-np.mean( time_sa[i+1]))/np.std(time_sa[i+1])
            
            mag_i = np.abs( norm_i )
            mag_i1 = np.abs( norm_i1 )
            
            ### get the standard deviation of the normalized aperture difference
            std_mag = std_filter( mag_i - mag_i1 )
            std_mag /= np.min( std_mag)
            bin_std_mag = np.zeros(std_mag.shape, dtype=np.uint8)
            bin_std_mag[std_mag > 50] = 1
            ### morphological close and fill in the holes
            closed = bs.close_hole(bin_std_mag)

            nlabels, labels, stats_main, centroids_main= cv2.connectedComponentsWithStats(closed)
            mean_val = np.zeros( std_mag.shape)
            for i in range(1, nlabels):
                rstart = stats_main[i, 1]
                cstart = stats_main[i,0]
                rend = rstart + stats_main[i,3]
                cend = cstart + stats_main[i,2]
                
                labels_chip = labels[rstart:rend, cstart:cend] 
                std_mag_chip = std_mag[rstart:rend, cstart:cend] 
                td_reg = self.time_deviation[rstart:rend, cstart:cend]  

                ireg = np.where( labels_chip== i )
                mean = np.mean( std_mag_chip[ireg])
                mean_val[ireg[0]+rstart, ireg[1]+cstart]  = mean

            imean = self.time_deviation < mean_val
            self.time_deviation[imean] = mean_val[imean]
            a = 1
        
        a = 1
                
            
            
        
        
        
        
        
        
        
        
        
        