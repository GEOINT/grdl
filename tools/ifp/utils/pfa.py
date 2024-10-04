# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:57:22 2024

@author: User
"""

from grdl.interpolation.polyphaseinterpolation import Polyphase_Interpolation

import numpy as np

import time

class PFA():

    def __init__(self, meta, geometry, polar_grid ):

        self.poly = Polyphase_Interpolation()

        self.poly.build_filter_function()

        self.meta = meta
        self.geometry = geometry
        self.polar_grid = polar_grid


    def interp_range(self, phd, pg):
        
        do_deskew = False
        if do_deskew:
            print( 'deskew started' )
            deskew = Deskew(self.meta, self.geometry, pg)
            phd = deskew.deskew( phd ).astype( np.complex64)
            print( 'deskew done')
        #rg_x_resample = np.linspace( pg.kv_bounds[0],pg.kv_bounds[1], pg.recNSamples)
        sampling = (pg.kv_bounds[1]-pg.kv_bounds[0])/pg.recNSamples
        rg_x_resample = np.arange( pg.recNSamples )*sampling + pg.kv_bounds[0]
        
        resamp_rg = np.zeros( (self.meta.npulses, pg.recNSamples ), dtype =phd.dtype )

        t0 = time.perf_counter()

        for i in range( self.meta.npulses ):

            kv = pg.return_polar_grid_samples(i)[1]

            #resamp_rg[i,:] = self.poly.poly_interp( pg.k_v[i,:], phd[i,:], self.poly.the_filter, rg_x_resample)

            resamp_rg[i,:] = self.poly.poly_interp( kv, phd[i,:], self.poly.the_filter, rg_x_resample)
            #resamp_rg[i,:] = np.interp( rg_x_resample, kv, phd[i,:])


        

        t1 = time.perf_counter()

        print( 'range sampling took: ', t1 - t0, ' s')


        return resamp_rg


    def interp_azimuth(self, ks, pg):

        az_x_resample = np.linspace( pg.ku_bounds[0], pg.ku_bounds[1], pg.recNPulses )
        sampling = (pg.ku_bounds[1]-pg.ku_bounds[0])/pg.recNPulses
        az_x_resample = np.arange( pg.recNPulses )*sampling + pg.ku_bounds[0]
        
        
        #flip = False

        resamp_az = np.zeros( (pg.recNPulses, pg.recNSamples ), dtype =ks.dtype )

        t0 = time.perf_counter()

        #kv_ks = np.linspace( pg.kv_bounds[0],pg.kv_bounds[1], pg.recNSamples)
     
        #rect_ss_rg = (pg.kv_bounds[1]-pg.kv_bounds[0])/pg.recNSamples
        #kv_ks = np.arange( pg.recNSamples )*rect_ss_rg + pg.kv_bounds[0]   
        proj = np.tan( pg.coord.phi )
        #proj = np.tan( self.geometry.poly_fit(self.meta.time) )
        kv_ks = np.linspace( pg.kv_bounds[0], pg.kv_bounds[1], pg.recNSamples )
        # if az_x_resample[0] > az_x_resample[-1]:
        #     flip = True
        #     proj = np.flip( proj )
        #     az_x_resample = np.flip( az_x_resample )
        #     ks = np.flip( ks, axis = 0)
        #     #az_x_resample = np.flip( az_x_resample)
        for i in range( pg.recNSamples ):
            
            ku_ks = kv_ks[i]*proj
            # if flip:
                
            #     resamp_az[:,i] = self.poly.poly_interp( np.flip(ku_ks), np.flip(ks[:,i]), self.poly.the_filter, np.flip(az_x_resample))

            # else:

            resamp_az[:,i] = self.poly.poly_interp( ku_ks, ks[:,i], self.poly.the_filter, az_x_resample)
            #resamp_az[:,i] = np.interp( az_x_resample, ku_ks, ks[:,i] )
            a = 1
        # if flip:
        #     resamp_az = np.flip( resamp_az, axis = 0 )
        t1 = time.perf_counter()

        #if az_x_resample[0] < az_x_resample[-1]:

        #    resamp_az = np.flip(resamp_az, axis = 1)


        print( 'azimuth sampling took: ', t1 - t0, ' s')

            #intFunc = sciint.interp1d( polar_grid.k_u_ks[:,i], resamp_rg[:,i])

        return resamp_az

    def perform_pfa(self, reader):
        
        showRaw = False
        if showRaw:
            from scipy import fft
            import matplotlib.pyplot as plt
            from sarpy.visualization.remap import Density
            rm = Density()
            dat = reader[(self.meta.first_valid_pulse, self.meta.full_npulses), (0, self.meta.nsamples)]
            plt.figure()
            plt.imshow( rm( dat ), cmap='gray', aspect= 'auto')
            plt.title('raw data')
        
        ks = self.interp_range(reader[(self.meta.first_valid_pulse, self.meta.full_npulses), (0, self.meta.nsamples)], self.polar_grid)
        
        #return ks.T
        #pfa = self.interp_azimuth( ks, self.polar_grid)
        
        # r1 = np.fft.fftshift( fft.fft2(dat) )
        # r2 = np.fft.fftshift( fft.fft2( ks) )
        # r3 = np.fft.fftshift(fft.fft2(pfa))
        
        

        
        # plt.figure()
        # plt.imshow( rm( r2 ), cmap='gray', aspect= 'auto')
        # plt.title('range interpolation data')
        # plt.figure()
        # plt.imshow( rm( r3 ), cmap='gray', aspect= 'auto')
        # plt.title('pfa data')
        return self.interp_azimuth( ks, self.polar_grid).T
        
