# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:21:20 2024

@author: User
"""
from sarpy.visualization.remap import Density
import matplotlib.pyplot as plt
import numpy as np

def plot_sar_image( im, remap = True, title='' ):
    
    plt.figure()
    
    if remap == True:
        rm = Density()
        plt.imshow(rm( im ), cmap='gray', aspect='auto')
    else:
        plt.imshow( np.abs( im ), cmap='gray', aspect='auto')
        
    plt.title(title)
        
        
        
        
        
        
        
        
        
        