# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:16:12 2024

@author: User
"""

from scipy import fft
import numpy as np

def compress_phd_data(phd):
    '''
    perform the frequency shifting and fft to take the data from 
    the phase history domain to the image domain with the proper shifting
    after polar formatting the data
    '''
    phd = np.fft.ifftshift( phd )
    
    return np.fft.fftshift( fft.ifft2( phd ) )
    
    