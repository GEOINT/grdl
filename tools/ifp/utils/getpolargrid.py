# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:34:21 2024

@author: User
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy 
class Get_Polar_Grid:
    '''
    Get the polar grid from the per vector parameters. This provides all the information
    for the intepolation grid. 
    
    Under the polar format algorithm the data is assumed to lie on a polar grid.
    The polar grid, in the slant plane, is determined by the aperature angle and 
    transmit frequencies. This then forms an annulus in spatial frequncy space.
    The polar format algorithm then needs to reshape points on the annulus into a 
    rectangular form so that a Fourier Transform can be performed to transform the
    data into cross sectional return space.
    
    First step is to set the foundational annulus where the data lies:
        set_polar grid()
    
    After this then calculations of the bounds and sampling rates of the data are 
    determined. 
    '''
    
    def __init__(self, meta, coord, range_comp=1, azimuth_comp= 1, inscription = True ):
        self.coord = coord
        self.meta = meta
        self.range_compression = range_comp
        self.azimuth_compression = azimuth_comp
        self.sf_conv = 2 / self.coord.c
        
        if inscription:
            self.grid_form = 'INSCRIBED'
        else:
            self.grid_form = 'CIRCUMSCRIBED'
        self.get_polar_grid_limits()

    def return_polar_grid_samples( self, 
                                   pulse_number:int ):
        '''
        this returns the polar grid kv values. this is the range direction
        which is dependent on the number of samples
        '''
        
        ##get the scaling factor for the projeciton, if slant it's 1
        scaling = self.coord.k_sf[pulse_number]*self.coord.vpmag
        #scaling = 1
        freq = np.arange( self.meta.nsamples )*self.meta.fxss[pulse_number] + self.meta.fx0[pulse_number]
        #freq = freq*( 1 + self.meta.aFDOP[pulse_number])
        
        spatial_freq = self.sf_conv*(freq)
        #freq = self.sf_conv*(np.arange( self.meta.nsamples )*self.meta.fx_sampling + self.meta.fx0[pulse_number])

        # if self.meta.chirprate < 0 :
        #     freq = np.flip(freq)
        
        #freq = self.sf_conv*np.linspace( self.meta.fx1[0], self.meta.fx2[0], self.meta.nsamples )

        #freq = self.sf_conv*(np.arange( self.meta.nsamples )*self.meta.fxss[pulse_number] + self.meta.fx1[pulse_number])
        #freq = self.sf_conv*(np.linspace( self.meta.fx1[pulse_number],self.meta.fx2[pulse_number], self.meta.nsamples) )

        
        ku = spatial_freq*np.sin( self.coord.phi[pulse_number] ) * scaling
        kv = spatial_freq*np.cos( self.coord.phi[pulse_number] ) * scaling
        #ku = spatial_freq*np.sin( self.coord.poly_fit(self.meta.time[pulse_number] )) / scaling
        #kv = spatial_freq*np.cos( self.coord.poly_fit(self.meta.time[pulse_number] )) / scaling        
        return ku, kv, freq
    
    def return_polar_grid_pulses( self, 
                                   sample_number:int ):
        '''
        return the polar grid ku values. This is the azimuth or cross range
        direction which is dependent on the number of pulses

        Parameters
        ----------
        sample_number : int
            DESCRIPTION.

        Returns
        -------
        ku : TYPE
            spatial frequency x direction (1/m).
        kv : TYPE
            spatial frequqncy y direction (1/m).
        freq : TYPE
            frequency values in Hz.

        '''
        ##get the scaling factor for the projeciton, if slant it's 1
        scaling = self.coord.k_sf*self.coord.vpmag
        #scaling = 1
        if sample_number == -1:
            sample_number = self.meta.nsamples-1
        freq = sample_number*self.meta.fxss+ self.meta.fx0
        #freq = freq*(1+ self.meta.aFDOP)
        spatial_freq = self.sf_conv*freq
        #freq = self.sf_conv*(sample_number*self.meta.fx_sampling+ self.meta.fx0)

        #ku = spatial_freq*np.sin( self.coord.poly_fit(self.meta.time) ) / scaling
        #kv = spatial_freq*np.cos( self.coord.poly_fit(self.meta.time)  ) / scaling
        ku = spatial_freq*np.sin( self.coord.phi ) * scaling
        kv = spatial_freq*np.cos( self.coord.phi ) * scaling              
        return ku, kv, freq    
    
    def build_full_grid( self ):

        freq = self.sf_conv*(np.reshape(self.meta.fx1, (self.meta.npulses,1)) +np.reshape( self.meta.fxss, (self.meta.npulses,1 ) )*np.reshape(np.arange( self.meta.nsamples),(1,self.meta.nsamples)))        

        scaling = np.reshape(self.coord.k_sf*self.coord.vpmag,(self.meta.npulses,1))
        
        ku = freq*np.reshape( np.sin(self.coord.phi ), (self.meta.npulses,1) )
        kv = freq*np.reshape( np.cos(self.coord.phi ), (self.meta.npulses,1) )

    def get_polar_grid_limits(self):

        scale = self.coord.k_sf
        sf_v_fx1 = self.sf_conv*self.meta.fx1*np.cos(self.coord.phi) * (scale)
        sf_v_fx2 = self.sf_conv*self.meta.fx2*np.cos(self.coord.phi) * (scale)
        
        nfreq = int( ( np.mean(self.meta.fx2)-np.mean(self.meta.fx1) ) / self.meta.fxss.mean() )
        bound_freq = np.linspace(self.meta.fx1.mean(), self.meta.fx2.mean(), nfreq )
        ### the u bounds at the first and last pulse
        sf_u_p0 = self.sf_conv*bound_freq*np.sin(self.coord.phi[0]) * (scale[0])
        sf_u_pN = self.sf_conv*bound_freq*np.sin(self.coord.phi[-1]) * (scale[-1])
        
        bw = (self.meta.fx2[0]-self.meta.fx1[0])
        print( f'bandwidth of collection: {bw*1e-6} MHz')
        
        if self.grid_form == 'INSCRIBED':
            kv_min_ind = np.argmax( np.abs( sf_v_fx1 ) )
            kv_max_ind = np.argmin( np.abs( sf_v_fx2 ) )
            kv_min = sf_v_fx1[kv_min_ind]
            kv_max = sf_v_fx2[kv_max_ind]

            self.kv_bounds = np.array([kv_min, kv_max])
            
            iumin = np.argmin( np.abs( sf_u_p0-sf_u_pN ) )
            ku_min = sf_u_p0[iumin]
            ku_max = sf_u_pN[iumin]
            
            #self.ku_bounds = np.array([self.kv_bounds[0]*np.tan( self.coord.phi[0]),self.kv_bounds[0]*np.tan( self.coord.phi[-1]) ])
            self.ku_bounds = np.array([ku_min,ku_max])
            a = 1
        elif self.grid_form == 'CIRCUMSCRIBED':
            
            kv_min_ind = np.argmin( np.abs( sf_v_fx1 ) )
            kv_max_ind = np.argmax( np.abs( sf_v_fx2 ) )
            kv_min = sf_v_fx1[kv_min_ind]
            kv_max = sf_v_fx2[kv_max_ind]

            self.kv_bounds = np.array([kv_min, kv_max])

            min_p0 = np.min( sf_u_p0 )
            max_p0 = np.max( sf_u_p0 )
            min_pN = np.min( sf_u_pN )
            max_pN = np.max( sf_u_pN )

            self.ku_bounds = np.array([min([min_p0, min_pN]), max([max_p0, max_pN])])
      
        
    def plot_polar_grid(self):
        pg_firstpulse = np.array( [self.return_polar_grid_samples(0)[0], self.return_polar_grid_samples(0)[1]] ).T
        pg_lastpulse = np.array( [self.return_polar_grid_samples(-1)[0], self.return_polar_grid_samples(-1)[1]] ).T    
        
        pg_firstsample = np.array( [self.return_polar_grid_pulses(0)[0], self.return_polar_grid_pulses(0)[1]] ).T
        pg_lastsample =np.array( [self.return_polar_grid_pulses(-1)[0], self.return_polar_grid_pulses(-1)[1]] ).T        
        
        plt.figure()
        plt.scatter( pg_firstpulse[:,0], pg_firstpulse[:,1])
        plt.scatter( pg_lastpulse[:,0], pg_lastpulse[:,1])
        plt.scatter( pg_firstsample[:,0], pg_firstsample[:,1])
        plt.scatter( pg_lastsample[:,0], pg_lastsample[:,1])
        
        plt.plot( [self.ku_bounds[0],self.ku_bounds[0]], [self.kv_bounds[0], self.kv_bounds[1]], color='r')
        plt.plot( [self.ku_bounds[0],self.ku_bounds[1]], [self.kv_bounds[0], self.kv_bounds[0]], color='g')
        plt.plot( [self.ku_bounds[1],self.ku_bounds[1]], [self.kv_bounds[1], self.kv_bounds[0]], color='r')
        plt.plot( [self.ku_bounds[1],self.ku_bounds[0]], [self.kv_bounds[1], self.kv_bounds[1]], color='g')        
    
        plt.xlabel('Cross Range Spatial Frequency (1/m)')
        plt.ylabel('Range Spatial Frequency (1/m)')
        plt.title('Polar Grid of the Collection')
    
    # def set_polar_grid(self):
    #     '''
    #     set the polar annulus that the data is collected on. This is determined 
    #     by the collection geometry and the received frequency.
    #     '''
        
    #     freq = np.arange( self.meta.nsamples )*self.meta.fx0 + self.meta.fxss
        
    #     ### this is the collection angle, with respect to the slant plane, per pulse
    #     self.phi = self.coord.phi 
        
        
    #     ### polar grid the data lies on
    #     k_u = np.zeros( (self.coord.npulses, self.coord.nsamples ))
    #     k_v = np.zeros( (self.coord.npulses, self.coord.nsamples ))
    #     for i in range( self.coord.npulses ):
    #         #freq = np.arange( self.meta.nsamples ) * self.meta.scss[i] + self.meta.sc0[i]
    #         k_u[i,:] = sf_conv*freq.flatten()*np.sin(self.coord.phi[i]) / (self.coord.k_sf[i]*self.coord.vpmag)
    #         k_v[i,:] = sf_conv*freq.flatten()*np.cos(self.coord.phi[i]) / (self.coord.k_sf[i]*self.coord.vpmag)

    def return_keystone_ku( self, kv_ks, sample_number):
        return kv_ks[sample_number]*np.tan( self.coord.phi)

    def polar_grid(self):

        self.proc_sf = np.abs( np.diff( self.kv_bounds ))
        self.procbw = np.abs((self.kv_bounds[1]-self.kv_bounds[0])/self.sf_conv  )      
        
        
        
        fullbw = self.meta.fx2[0] - self.meta.fx1[0]

        ###range resolution, spatial domain, of processed bandwidth
        rho_r = self.coord.c*0.88/ (2*self.procbw)*self.range_compression
        sf_range = 2/rho_r
        ###range resolution, spatial frequency domain
        
        range_freq_sampling =  1/np.abs(2*self.meta.reader.cphd_meta.TxRcv.TxWFParameters[0].LFMRate / (self.coord.c*self.meta.fad))*self.range_compression
        nyquist_freq_sampling = 2/self.meta.rcv_window/0.88*self.range_compression
        range_sf_sampling =  self.meta.fxss[0]*self.sf_conv
        #Ny = int( (self.kv_bounds[1]-self.kv_bounds[0])/ range_sf_sampling )
        range_sf_sampling = nyquist_freq_sampling*self.sf_conv
        Ny = int( self.procbw / nyquist_freq_sampling)
        
        
        bw_compression = self.procbw/fullbw
        range_sf_resolution = bw_compression*1/self.meta.fxss[0]
        centerFreq = np.mean(self.kv_bounds)/self.sf_conv
        centerWL = np.mean( self.coord.c/centerFreq )
        self.centerWL = centerWL
        rho_a = centerWL*0.88 / (4*np.sin( self.coord.Theta / 2) )*self.azimuth_compression
        
        
        range_resolution = self.meta.c*0.89 / (2*self.procbw)
        azimuth_resolution = centerWL / (2*self.coord.Theta)*self.azimuth_compression
        self.azimuth_resolution = azimuth_resolution
        self.range_resolution = range_resolution
        print( f'range_resolution: {range_resolution}, azimuth resolution: {azimuth_resolution}')
        ### get the azimuth spatial frequency sampling
        ## using the keystone grid, find the lowest sample, get absolute difference
        ## of the samples and then find the closest spaced azimuth samples.
        azimuth_sf_sampling = np.min(np.abs(np.diff(self.kv_bounds[0]*np.tan( self.coord.phi ) ) ) )*0.889*self.azimuth_compression

        print( 'centerWL: ', centerWL, azimuth_resolution, azimuth_sf_sampling )

        #Ny = int(np.floor((self.kv_bounds[1] - self.kv_bounds[0])/range_sf_resolution))
        ku_span = np.abs( np.diff( self.ku_bounds ))
        Nx = int(ku_span/azimuth_sf_sampling)


        self.recNPulses = Nx
        self.recNSamples = Ny
        npulses = self.coord.npulses
        nsamples = self.coord.nsamples
        print( 'starting point: ', npulses, nsamples )


        # if self.kv_bounds[0] > self.kv_bounds[1]:
        #     self.kv_bounds = np.flip(self.kv_bounds)
       
        ### next make the keystone grid
        #kv_ks = np.linspace(kv_bounds[0], kv_bounds[1], self.recNSamples)
        kv_ks = np.linspace(self.kv_bounds[0], self.kv_bounds[1], self.recNSamples)

        # k_v_ks = np.zeros( (npulses, self.recNSamples))
        # k_u_ks = np.zeros( (npulses, self.recNSamples))
        # for i in range( npulses ):
        #     k_v_ks[i,:] = kv_ks
        #     k_u_ks[i,:] = kv_ks*np.tan( self.coord.phi[i])

        self.rg_impresbw = self.kv_bounds[1] - self.kv_bounds[0]
        self.rg_impreswid = 0.886 / self.rg_impresbw
        self.rg_ss =1/(self.rg_impresbw)
        self.rg_kctr = np.mean( self.kv_bounds )
        self.rg_deltak1 = self.kv_bounds[0]
        self.rg_deltak2 = self.kv_bounds[1]
        
        self.az_impresbw = np.abs( np.diff( self.ku_bounds))
        self.az_impreswid = 0.886 / self.az_impresbw
        self.az_ss = 1/ (self.az_impresbw)
        self.az_kctr = np.mean( self.ku_bounds)
        #self.az_deltak1 = self.ku_bounds[self.ku_bounds>0]
        #self.az_deltak2 = self.ku_bounds[self.ku_bounds<0]
        #k_x = np.linspace( ku_bounds[0], ku_bounds[1], self.recNPulses )
        k_x = np.linspace( self.ku_bounds[0], self.ku_bounds[1], self.recNPulses )


        print( 'Sampling: ', '\n')
        print( npulses, Nx, nsamples, Ny )
        print( self.ku_bounds )
        print( self.kv_bounds )

        #self.k_v = k_v
        # self.k_u_ks = k_u_ks
        self.k_x = k_x

    def build_sicd_meta(self, sicd_build):
        

        
        
        
        
        a = 1