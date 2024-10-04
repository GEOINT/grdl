# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 07:30:54 2024

@author: User
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy import stats
import matplotlib.pyplot as plt

class Binary_Segs():
    
    def __init__( self ):
        a = 1
        
        self.closing_kernel = (3,3) 
    
        
    
    def close_hole(self, binary_inimage, closing_kernel= None):
        '''
        Parameters
        ----------
        inimage : ndarray
            input image that is to be segmented. this is to be a binary image
        close_kernel: list
            closing kernel to be used with the morphological close operation

        Returns
        -------
        closed_hole : ndarray
            image with a closed and holes closed

        '''
        if closing_kernel == None:
            closing_kernel = self.closing_kernel

        
        close_hole = np.zeros( binary_inimage.shape, dtype=np.uint8)

        ### I want to now take each region, segment them, close them, and replace 
        ## the region pixel value with the median value
        nlabels, labels, stats_main, centroids_main= cv2.connectedComponentsWithStats(binary_inimage)
        
        krsize = int(np.array( closing_kernel[0], dtype = int))
        kcsize = int(np.array( closing_kernel[1], dtype = int))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kcsize,krsize))
        debug = False
        for i in range( 1, nlabels):
            ### if the region is greater than 1 pixel area
            if stats_main[i,4] >1:
                rstart = stats_main[i, 1]
                cstart = stats_main[i,0]
                rend = rstart + stats_main[i,3]
                cend = cstart + stats_main[i,2]
                
                chip_reg = binary_inimage[rstart:rend, cstart:cend]
                chip_label = labels[rstart:rend, cstart:cend]

                bin_reg = np.zeros( chip_reg.shape)

                ireg = chip_label == i
                bin_reg[ireg] = 1
                ### close the region
                c_reg = cv2.morphologyEx( bin_reg, cv2.MORPH_CLOSE, kernel, iterations=1 )       
                
                ### fill the holes
                cfill_ = np.zeros( c_reg.shape, dtype=np.uint8)
                contour,hier = cv2.findContours(c_reg.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                for c_ndx in range(len(contour)):
                    if hier[0][c_ndx][3] > -1:
                        cv2.drawContours(cfill_, [contour[c_ndx]], 0, 255, -1)
                    
                c_reg[cfill_>0] =  1
                
                ### get the closed region pixels
                ireg = c_reg == 1
                ### get the statistics of the input
                #stat = stats.describe( chip_reg[ireg])
                #median = np.median( chip_reg[ireg])
                #mean = np.mean( chip_reg[ireg])
                fill_ndx = np.where( c_reg > 0 )
                
                close_hole[fill_ndx[0]+rstart, fill_ndx[1]+cstart] = 1
                
                if debug:
                    fig1 = plt.figure()
                    plt.imshow( chip_reg, aspect='auto', cmap ='gray' )
                    plt.title(f'chipped region std_mag reg: {i}')
                    fig2 = plt.figure()
                    plt.imshow( bin_reg, aspect='auto', cmap = 'gray', vmin = 0, vmax =1)
                    plt.title(f'chipped binary input reg: {i}')
                    fig3 = plt.figure()
                    plt.imshow( c_reg, aspect='auto', cmap='gray', vmin = 0, vmax = 1)
                    plt.title(f'chipped binary close reg: {i}')
                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig3)
                
                a = 1
            
        return close_hole