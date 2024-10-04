# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:13:25 2024

@author: User
"""

import cv2
import numpy as np

def std_filter( inimage, size = (5,5) ):
    mean = cv2.blur( inimage, size)
    mean_sq = cv2.blur( inimage**2, size )
    
    return np.sqrt( mean_sq - mean**2 )