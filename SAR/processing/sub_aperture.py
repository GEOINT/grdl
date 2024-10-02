# -*- coding: utf-8 -*-
"""
Created on Monday Sep 30 13:01:37 2024

@author: User
"""

import numpy as np
from SAR.processing.phd_return import PHD_Return

class sub_aperture():
    '''
    a class to sub-aperture SAR SICD data. The time sub-aperture is in the cross-range 
    direction or the columns the frequency sub-aperture is in the range direction or the
    rows
    
    example run time:
        reader = SICDReader(file_name)
        meta = reader.sicd_meta
        dat_chip = reader[rstart:rend, cstart:cend]

        sa = sub_aperture(meta)

        sa_time = sa.get_time_sub_aperture(dat_chip)

        plt.figure()
        plt.imshow(rm(dat_chip), cmap='gray', aspect='auto')
        plt.title('image data')

        plt.figure()
        plt.imshow(rm(sa_time[0]), cmap='gray', aspect='auto')
        plt.title('early time sub aperture')


        plt.figure()
        plt.imshow(rm(sa_time[-1]), cmap='gray', aspect='auto')
        plt.title('late time sub aperture')
    
    '''
    def __init__(self, meta):
        self.meta = meta
        self.phd_r = PHD_Return(self.meta)
        
    def get_time_sub_aperture(self, chip, num_aps = 4, overlap = .0):
        nrows, ncols = chip.shape
        
        phd_pix_range = self.phd_r.get_phd_data_pixel_range((nrows,ncols))
        phd_pix_start = [phd_pix_range[0], phd_pix_range[2]]
        dat_width = ncols - 2*phd_pix_start[1]
        
        division_size = dat_width/num_aps
        ap_size = dat_width/num_aps*(1+overlap)
        overlap_size = division_size*overlap
        rstart = phd_pix_start[1] + (division_size)*np.arange(num_aps) -overlap_size
        rstart = np.round(rstart).astype(int)
        sub_aps = np.zeros( (num_aps, nrows, ncols ), dtype = chip.dtype )
        
        for i in range( num_aps ):
            phd_chip = np.fft.fftshift(np.fft.fft(chip, axis = 1 ) )
            
            phd_chip[:,0:rstart[i]] = 0
            phd_chip[:,rstart[i]+ap_size.astype(int):] = 0
            
            sub_aps[i] = np.fft.ifft( np.fft.ifftshift(phd_chip), axis = 1 )
        
        return sub_aps
    
    def get_freq_sub_aperture(self, chip, num_aps = 4, overlap = .0):
        nrows, ncols = chip.shape
        
        phd_pix_range = self.phd_r.get_phd_data_pixel_range((nrows,ncols))
        phd_pix_start = [phd_pix_range[0], phd_pix_range[2]]
        dat_width = ncols - 2*phd_pix_start[0]
        
        division_size = dat_width/num_aps
        ap_size = dat_width/num_aps*(1+overlap)
        overlap_size = division_size*overlap
        cstart = phd_pix_start[0] + (division_size)*np.arange(num_aps) -overlap_size
        cstart = np.round(cstart).astype(int)
        sub_aps = np.zeros( (num_aps, nrows, ncols ), dtype = chip.dtype )
        
        for i in range( num_aps ):
            phd_chip = np.fft.fftshift(np.fft.fft(chip, axis = 0 ) )
            
            phd_chip[0:cstart[i], :] = 0
            phd_chip[cstart[i]+ap_size.astype(int):,:] = 0
            
            sub_aps[i] = np.fft.ifft( np.fft.ifftshift(phd_chip), axis = 0 )
        
        return sub_aps
        a = 1
        
        
        
        
'''
dat_chip = reader[rstart:rend, cstart:cend]

sa = sub_aperture(meta)

sa_time = sa.get_time_sub_aperture(dat_chip)

plt.figure()
plt.imshow(rm(dat_chip), cmap='gray', aspect='auto')
plt.title('image data')

plt.figure()
plt.imshow(rm(sa_time[0]), cmap='gray', aspect='auto')
plt.title('early time sub aperture')


plt.figure()
plt.imshow(rm(sa_time[-1]), cmap='gray', aspect='auto')
plt.title('late time sub aperture')
'''     
        
        
        
        
        
        
        
        
        
        
        