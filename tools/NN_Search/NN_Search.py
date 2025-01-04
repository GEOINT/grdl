# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:42:38 2024

@author: User
"""
from scipy import ndimage
import numpy as np
from numba import njit, complex64, float64, float32
import cv2
class NearestNeighbor_Search():
    
    def __init__(self, shape):
        nrows, ncols = shape
        
        #### this is the distance metric of the image starting from the top left
        ### min row min col
        ### all other pixel values are just a simple radial translation from the 
        ### starting point
        self.xx, self.yy  = np.meshgrid(np.arange( ncols), np.arange(nrows))


    def calc_neighbor_distance( self, image, ndx ):
        import matplotlib.pyplot as plt
        from scipy import  stats
        
        binary = np.zeros( image.shape )
        binary[ndx] = 1
        
        # fig2, ax2 = plt.subplots( 1,2, sharex = True, sharey = True)
        # ax2[0].imshow( image, vmax = 150, cmap = 'gray', aspect='auto' )
        # ax2[0].set_title('chip mixed')

        # ax2[1].imshow( binary, vmin = 0, vmax = 1, aspect='auto', cmap='gray')
        # ax2[1].set_title('binary image')
        
        reg_stats = stats.describe( image[ndx] )
        
        adaptive_grow = binary.copy()
        Pass = False

        for i in range( len( ndx[0] ) ):
            pixel = [ ndx[0][i], ndx[1][i] ]
            distance = np.sqrt((self.xx- pixel[1])**2 + (self.yy - pixel[0])**2 )
            idis = ( distance < 3 ) & (distance >0)
            ibinary = adaptive_grow >0
            
            inn = idis & ~ibinary
            arg_true = np.argwhere( inn == True )
            
            n_points = arg_true.shape[0]
            if n_points > 0:
            
                nn_stats = stats.describe( image[inn] )
                

                if nn_stats.mean > .1*reg_stats.mean:
                    adaptive_grow[inn] = 1
                    Pass = True
                    
                # fig1, ax1 = plt.subplots( 1,2, sharex = True, sharey = True)
                # ax1[0].imshow( binary, vmax = 1, cmap = 'gray', aspect='auto' )
                # ax1[0].set_title(f'binary, {Pass}')
                # binary_nn = np.zeros( image.shape )
                # binary_nn[inn] = 1
                # ax1[1].imshow( binary_nn, vmin = 0, vmax = 1, aspect='auto', cmap='gray')
                # ax1[1].set_title(f'binary nn, {Pass}')
                
                # a =1
                # plt.close( fig1 )
                
        # if Pass:
        #     fig1, ax1 = plt.subplots( 1,3, sharex = True, sharey = True)
        #     ax1[0].imshow( binary, vmax = 1, cmap = 'gray', aspect='auto' )
        #     ax1[0].set_title(f'binary')
        #     binary_nn = np.zeros( image.shape )
        #     binary_nn[inn] = 1
        #     ax1[1].imshow( adaptive_grow, vmin = 0, vmax = 1, aspect='auto', cmap='gray')
        #     ax1[1].set_title(f'adaptive grow {Pass}')
        #     ax1[2].imshow( image, vmax = 50, aspect='auto', cmap='gray')
        #     ax1[2].set_title(f'mixed {Pass}')
        #     a = 1
        #     plt.close( fig1 )
        # plt.close(fig2)
        if Pass == True:
            a = 1
        return adaptive_grow
        
    def calc_neighbor_distance_morph( self, image, ndx ):
        import matplotlib.pyplot as plt
        from scipy import  stats
        
        binary = np.zeros( image.shape, dtype = np.uint8 )
        binary[ndx] = 1
        reg_stats = stats.describe( image[ndx] )
        kernel = ndimage.morphology.generate_binary_structure(2,1)
        kernel = ndimage.iterate_structure(kernel, 2)
        kernel_cv = kernel.astype(float)/np.nonzero(kernel)[0].size 
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        
        grown_morph_bin = np.zeros( binary.shape ) 
        grown_morph_bin[binary>0] = 1
        Pass = True
        while True:
            reg_mean =np.mean( image[grown_morph_bin>0])
            #dilate = ndimage.binary_dilation(grown_morph_bin, np.ones( (5,5) ) )
            dilate = cv2.dilate( grown_morph_bin, np.ones( (5,5) ))
            grown = dilate - grown_morph_bin
            
            mag_grown = np.zeros( grown.shape )
            mag_grown[grown>0] = image[grown > 0]
            

            
            #mean_filter = ndimage.generic_filter(mag_grown, np.mean, footprint =kernel)
            mean_filter = cv2.filter2D( mag_grown,-1, kernel_cv)
            binary_mean = np.zeros( mean_filter.shape )
            binary_mean[mean_filter> 0.1*reg_mean]  =1 
            
            if True in binary_mean:
                
                binary_grow = ndimage.binary_dilation(binary_mean, kernel)
    
                grown_binary = np.zeros( binary.shape )
                grown_morph_bin[(binary_grow>0)|(binary>0)] = 1
                a = 1
            else:
                break
        a = 1
        return grown_morph_bin

        
# import matplotlib.pyplot as plt

# arr = np.ones( (100,200) )

# nn = NearestNeighbor_Search(arr.shape)
# nn.calc_neighbor_distance((50,100))

# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# xx, yy = np.meshgrid(np.arange( arr.shape[1]), np.arange( arr.shape[0] ))
# ax.plot_surface( xx, yy, nn.distance)



        
        
        
        
        
        
        