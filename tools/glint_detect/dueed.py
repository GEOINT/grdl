# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 16:55:34 2024

@author: User
"""

from grdl.SAR.processing.sub_aperture import sub_aperture
from grdl.SAR.processing.mix_images import mix_images
from sarpy.visualization.remap import Density
from scipy import ndimage
import numpy as np
import cv2
from skimage.segmentation import flood, flood_fill
import matplotlib.pyplot as plt
import scipy.ndimage
import time
from grdl.tools.NN_Search.NN_Search import NearestNeighbor_Search

def plot_image( im ):
    plt.figure()
    plt.imshow( np.abs(im), aspect='auto', cmap='gray' )

class DUEED():
    def __init__(self, meta, num_aps = 3, overlap = 0.1):
        self.num_aps = num_aps
        self.overlap = overlap
        self.meta = meta
        self.rm = Density()
        
    def get_mixed_aps(self, dat):
        
        apy = sub_aperture(self.meta)
        t_ap = apy.get_time_sub_aperture(dat, self.num_aps, self.overlap)
        t_mixed = mix_images(t_ap)
        f_ap = apy.get_freq_sub_aperture(dat, self.num_aps, self.overlap)
        f_mixed = mix_images(f_ap)

        f_t_mixed = mix_images(np.array([t_mixed, f_mixed]))        
        pg = np.gradient( np.angle(dat), axis = 1)
        pg_mix = np.gradient( np.angle( f_t_mixed), axis = 1 )
        self.mixed = np.abs(f_t_mixed)
        #self.mixed = f_t_mixed.real+f_t_mixed.imag
        self.dat = dat
        
    def get_binary_regions(self, global_thresh = 100, local_thresh = 10):
        grad_thresh = 5
        
        debug = False
        nrows, ncols = self.mixed.shape
        binary_det = np.zeros( self.mixed.shape, dtype = np.uint8)
        
        binary_global = np.zeros( self.mixed.shape, dtype =np.uint8 )
        binary_global[self.mixed > global_thresh] = 1
        
        gradient = ndimage.sobel( self.mixed)
        gradient_mag = ndimage.sobel( np.abs(self.dat))

        rsize = int(np.array( 30 / self.meta.Grid.Row.SS, dtype = int))
        csize = int(np.array( 30 / self.meta.Grid.Col.SS, dtype = int))
        
        krsize = int(np.array( 5 / self.meta.Grid.Row.SS, dtype = int))
        kcsize = int(np.array( 5 / self.meta.Grid.Col.SS, dtype = int))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kcsize,krsize))

        if debug:
            fig, ax = plt.subplots(1 , 2, sharex = True, sharey = True)
            ax[0].imshow( self.mixed, aspect='auto', cmap='gray', vmax = 50)
            ax[0].set_title('mixed')
            ax[1].imshow( binary_global, aspect='auto', cmap= 'gray', vmax = 1)
            ax[1].set_title('binary global')

        nlabels, labels, stats_main, centroids_main= cv2.connectedComponentsWithStats(binary_global)
        
        for i in range( 1, nlabels):
        #for i in range( 1008, nlabels):
            cval = stats_main[i,0].copy()
            rval = stats_main[i,1].copy()
            if binary_det[rval, cval] == 0:
                rstart = stats_main[i,1]
                rstart -= rsize//2
                rend = stats_main[i,1] + stats_main[i,3] + rsize//2
                if rstart < 0:
                    rend = rend - rstart
                    rstart = 0
                if rend > nrows:
                    rstart = rstart - (rend-nrows)
                    rend = nrows
                    
                cstart = stats_main[i,0]
                cstart -= csize//2
                cend = stats_main[i,0]+ stats_main[i,2] + csize//2
                if cstart < 0:
                    cend = cend - cstart
                    cstart = 0
                if cend > ncols:
                    cstart = cstart - (cend-ncols)
                    cend = ncols
                
                cval = stats_main[i,0] - cstart
                rval = stats_main[i,1] - rstart
                print( 'rval, cval: ', rval, cval)
                chip_det = binary_det[rstart:rend, cstart:cend].copy()
                chip_bin = binary_global[rstart:rend, cstart:cend].copy()
                chip_mag = self.mixed[rstart:rend, cstart:cend].copy()
                chip_dat = self.dat[rstart:rend, cstart:cend].copy()
                chip_grad = np.abs(gradient[rstart:rend, cstart:cend].copy())
                chip_label = labels[rstart:rend, cstart:cend].copy()
                
                chip_label[chip_label != i] = 0
                chip_bin_grad = np.zeros( chip_grad.shape, dtype = np.uint8)
                chip_bin_grad[chip_grad >grad_thresh] = 1
                chip_bin_grad[rval,cval] = 1
                ### segment region out of this
                retval_grad, labels_grad, stats_grad, centroids_grad= cv2.connectedComponentsWithStats(chip_bin_grad.astype(np.uint8))
                reg_val = labels_grad[rval, cval]
                
                if reg_val > 0:
                    chip_grad_reg = np.zeros( chip_grad.shape, dtype = np.uint8)
                    chip_grad_reg[labels_grad == reg_val] = 1
                                    
                    cap_chip_mag = chip_mag.copy()
                    cap_chip_mag[chip_mag > global_thresh] = global_thresh
                    #chip_flood = flood_fill( cap_chip_mag, (rval,cval), 150, tolerance=1)
                   
                    nn = NearestNeighbor_Search(chip_bin.shape)
                    ndx = np.where( chip_grad_reg > 0 )
                    #ndx = np.where( chip_label > 0)
                    #c_reg = cv2.morphologyEx( chip_grad_reg, cv2.MORPH_CLOSE, kernel )   
                    t0 = time.perf_counter()
                    adaptive_grow = nn.calc_neighbor_distance_morph(chip_mag, ndx)
                    t1 = time.perf_counter()
                    #print( f'morph grow: {t1-t0}')
                    # t0 = time.perf_counter()
                    # adaptive_grow = nn.calc_neighbor_distance(chip_mag, ndx)
                    # t1 = time.perf_counter()
                    #print( f'brute_force way: {t1-t0}')
                    a = 1
                    
                    c_reg = ndimage.binary_fill_holes(adaptive_grow )
                    if debug:


                        fig1, ax1 = plt.subplots( 1,2, sharex = True, sharey = True)
                        ax1[0].imshow( chip_bin, vmax = 1, cmap = 'gray', aspect='auto' )
                        ax1[0].set_title('chip bin')
                        ax1[1].imshow( chip_mag, vmax = 50,vmin = 0, cmap='gray', aspect='auto')
                        ax1[1].set_title('chip_mag')
                        
                        fig2, ax2 = plt.subplots( 1,2, sharex = True, sharey = True)
                        ax2[0].imshow( chip_grad, vmax = 15, cmap = 'gray', aspect='auto' )
                        ax2[0].set_title('chip grad')
                        ax2[1].imshow( chip_grad_reg, vmin = 0, vmax = 1, aspect='auto', cmap='gray')
                        ax2[1].set_title('chip bin grad region')
                        
                        fig3, ax3 = plt.subplots( 1,2, sharex = True, sharey = True)
                        ax3[0].imshow( self.rm(chip_dat), vmax =150,  cmap = 'gray', aspect='auto' )
                        ax3[0].set_title('chip dat')
                        ax3[1].imshow( c_reg, vmin = 0, vmax = 1, aspect='auto', cmap='gray')
                        ax3[1].set_title('c_reg morph close')
                        a = 1
                        
                        plt.close(fig1)
                        plt.close( fig2)
                        plt.close(fig3)
                    # nlab, labs, stats, centroids = cv2.connectedComponentsWithStats(chip_bin_grad)
                    # for j in range( 1, nlab):
                    #     bin_reg = labs == j
                    
                    
                
                    #ireg = chip_grad_reg > 0
                    
                    #if np.sum(chip_det[ireg]) == 0:
                        
        
                    #c_reg = cv2.dilate( chip_bin, kernel )       
            
                    binary_det[rstart:rend, cstart:cend] += c_reg



        
        self.binary_det = binary_det
        retval, labels_main, stats_main, centroids_main= cv2.connectedComponentsWithStats(binary_det.astype(np.uint8))

        center_points = centroids_main[1:]
        center_points = np.flip( centroids_main, axis = 1)
        self.center_points = center_points
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        