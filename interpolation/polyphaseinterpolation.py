# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:10:17 2024

@author: User
"""

import numpy as np

from scipy import signal

from numba import njit, complex64, float64, float32

import time

import matplotlib.pyplot as plt


class Polyphase_Interpolation():

    def __init__(self):
        self.n_taps = 101
        self.num_phases = 2048
        self.oversample = 1.25
        self.filter_length = 16
        self.filter_min_oversample = 0.05


    def build_filter_function(self):

        PASSBAND = 0.5 / self.oversample

        STOPBAND = 1 - PASSBAND

        full_filter_length = self.num_phases*self.filter_length +1

        scale_factor = full_filter_length/self.n_taps

        #scale_factor = self.num_phases/self.filter_length
        full_passband = PASSBAND / self.num_phases
        full_stopband = STOPBAND / self.num_phases
        bands = [0, full_passband*scale_factor, full_stopband*scale_factor, 0.5 ]
        kernel = signal.remez( self.n_taps, bands, desired=[1,0], weight=[self.filter_min_oversample, 1.0])

        # ### old, 1000

        # ### new, 1250

        # ### old bw, 700

        # fs_old = 2000

        # fs_new = 1250*2

        # bw = 300

        # fstop = (fs_new - bw) / fs_old

        # fpass = bw / fs_old

        # N_taps = int(60 / (22*(fstop-fpass)))

        # sf = fstop/ 0.5
        # fpass = fpass 
        # fstop = fstop
        # bands = [0,fpass, fstop,0.5]

        # kernel = signal.remez( N_taps, bands, desired=[1,0], weight=[self.filter_min_oversample, 1.0])

        full_kernel = np.fft.ifftshift( signal.resample( np.fft.fftshift(kernel), full_filter_length))
        full_kernel /= np.sum(full_kernel)

        col_indices = (np.arange( self.filter_length ) + 1)*self.num_phases
        row_indices = np.arange( self.num_phases+1)*-1

        filter_indices = (np.expand_dims(row_indices, axis = 1) + np.expand_dims(col_indices, axis = 0))
        the_filter = full_kernel[filter_indices].astype(np.float32)
        norm = np.expand_dims( the_filter.sum(axis=1), axis = 1)

        self.the_filter = the_filter/norm
    @staticmethod
    @njit(complex64[:](float64[:], complex64[:], float32[:,:], float64[:]))
    def poly_interp( xin, yin, poly_bank, xout ):

        sampling = np.abs(xout[1]-xout[0])
        phase_length, f_length = poly_bank.shape

        phase_length -= 1
        transform = [xout[0], 1/sampling]
        yout = np.zeros( len( xout ), dtype=yin.dtype )

        norm = np.zeros( len(xout ) )
        half_length = f_length//2-1


        for ndx, val in enumerate( yin ):

            out_ndx = (xin[ndx]-transform[0])*transform[1]
            base_ndx = int(out_ndx )

            if base_ndx - half_length > len(xout):
                continue

            fractional_offset = out_ndx - float( base_ndx )
            phasing = int(fractional_offset*float(phase_length)+0.5)
            start_ndx = base_ndx - half_length

            start_ndx = max( start_ndx, 0 )

            end_ndx = start_ndx + f_length
            end_ndx = min( end_ndx, len( xout) )
            filter_start = start_ndx - (base_ndx-half_length)
            filter_stop = filter_start + end_ndx - start_ndx

            for f_ndx, coef in enumerate(poly_bank[phasing, filter_start:filter_stop]):

                #if base_ndx < len(xout):

                yout[f_ndx + start_ndx] += coef*val

                norm[f_ndx + start_ndx] += coef


        for ndx in range( len(yout ) ):

           if norm[ndx] > 0.02:

                yout[ndx] /= norm[ndx]

           else:

                yout[ndx] = 0

        return yout

    '''

    @staticmethod

    @njit(complex128[:](float64[:], complex128[:], float64[:,:], float64[:]))

    def interpolate(xin, yin, filter, xout):

        yout = np.zeros( len(xout), dtype=yin.dtype )

        new_sampling = xout[1]-xout[0]

        norm = np.zeros( len( xout ), dtype = yin.dtype)


        print( new_sampling, xout[0] )

        for ndx, val in enumerate( yin ):

            out_ndx = (xin[ndx]-xout[0] )/new_sampling


            base_ndx = np.round( out_ndx ).astype(int)



            #for

                #yout[base_ndx] += (coef*yin)

            if ndx < 250:

                print( ndx, xin[ndx], out_ndx)


        return yout

    
    '''

def test_func():

    print( 'testing')



    alpha = 250


    #### test signal that samples a lfm

    sampling_freq = 5e5


    xin =np.arange( 0,sampling_freq)/sampling_freq

    input_signal = np.exp( 1j*2*np.pi*alpha*(xin)**2)


    sampling_out = 2.123e5

    xout = np.arange( 0, sampling_out)/sampling_out



    poly = Polyphase_Interpolation()

    poly.build_filter_function()


    t0 = time.perf_counter()

    yout = poly.poly_interp(xin, input_signal, poly.the_filter, xout )

    t1 = time.perf_counter()


    print( 'interpolation took: ', t1 - t0, ' s')


    plt.figure()

    plt.plot( xin, np.real(input_signal) )


    plt.plot( xout, yout)


    in_freq = (np.arange( len(xin ) )/len(xin) - 0.5)*sampling_freq

    in_spec = np.fft.fftshift( np.fft.fft( input_signal ) )


    out_len = 2*len(xout)

    out_freq = (np.arange( out_len )/out_len- 0.5)*sampling_out

    out_spec = np.fft.fftshift( np.fft.fft( yout, n = 2*len( yout) ),  )


    plt.figure()

    plt.plot( in_freq, np.abs(in_spec))

    plt.plot( out_freq, np.abs(out_spec) )



    plt.show()


#test_func()