# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:29:35 2024

@author: User
"""


from sarpy.io.phase_history.cphd import CPHDReader

import numpy as np
from numpy.linalg import norm
import matplotlib

#['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

#matplotlib.use('QtAgg')  # or 'GTK3Cairo'

import matplotlib.pyplot as plt


###example filename ='/Users/duanesmalley/SAR_Data/2023-07-26-02-23-35_UMBRA-05.cphd'

#filename = '/Users/duanesmalley/SAR_Data/2023-07-24-02-50-27_UMBRA-05.cphd'

class Meta_Setup():



    def __init__(self, filename):

        self.filename = filename

        self.reader = CPHDReader( self.filename )

        

        self.__setup()


    def __setup(self):

        meta = self.reader.cphd_meta
        self.cphdmeta = meta
        fx_Band = meta.Global.FxBand.get_array()
        bw = fx_Band[1] - fx_Band[0]
        self.rcv_window = meta.TxRcv.RcvParameters[0].WindowLength
        nchannels = meta.Data.NumCPHDChannels

        channel = 0
        self.chirprate = np.abs(meta.TxRcv.TxWFParameters[channel].LFMRate)
        c = 299792458.0

        self.c = c
        
        ##for now just process the first channel

        npulses = meta.Data.Channels[channel].NumVectors

        #npulses = int(npulses*.95)


        nsamples = meta.Data.Channels[channel].NumSamples
        #nsamples = 50000
        print( 'npulses: ', npulses)

        pvp = self.reader.read_pvp_array(channel)

        ### fill all the np arrays for the pvp

        self.__fill_pvp_data(meta, pvp)

        self.full_npulses = npulses

        ###first valid pulse, when signal >0

        self.first_valid_pulse = np.min( np.where( self.signal > 0) )
        #self.first_valid_pulse = 50

        self.npulses = npulses - self.first_valid_pulse
        
        print( 'meta self.npulses: ', self.npulses, meta.Data.Channels[channel].NumVectors, self.first_valid_pulse)

        self.nsamples = nsamples
        print( 'self.first_valid_pulse: ', self.first_valid_pulse)

        # if self.first_valid_pulse > 0:

        ind = self.first_valid_pulse
        self.txtime = self.txtime[ind:ind+self.npulses]
        self.txvel = self.txvel[ind:ind+self.npulses,:]
        self.txpos = self.txpos[ind:ind+self.npulses,:]
        self.rcvtime = self.rcvtime[ind:ind+self.npulses]
        self.rcvvel = self.rcvvel[ind:ind+self.npulses,:]
        self.rcvpos = self.rcvpos[ind:ind+self.npulses,:]
        self.scp = self.scp[ind:ind+self.npulses,:]
        self.fx1 = self.fx1[ind:ind+self.npulses]
        self.fx2 = self.fx2[ind:ind+self.npulses]
        self.fx0 = self.fx0[ind:ind+self.npulses]
        self.fxss = self.fxss[ind:ind+self.npulses]
        self.fad = np.mean( self.fxss )
        self.time = 1/2*( self.rcvtime + self.txtime)
        self.fx_sampling = np.mean( self.fxss )
        a = 1
        
    def __fill_pvp_data(self, meta, pvp):

        '''

        in general what I want to do is have a json input for

        names of the pvp data for the data set. Until then

        I've looked at the pvp and got the names. For

        cphd data the meta.PVP will contain all the information

        about the names and types of the meta data

        '''

        self.txtime = self.__set_pvp_data( meta, pvp, 'TxTime' )

        self.txvel = self.__set_pvp_data( meta, pvp, 'TxVel')

        self.txpos = self.__set_pvp_data( meta, pvp, 'TxPos')


        self.rcvtime = self.__set_pvp_data( meta, pvp, 'RcvTime')

        self.rcvvel = self.__set_pvp_data( meta, pvp, 'RcvVel' )

        self.rcvpos = self.__set_pvp_data( meta, pvp, 'RcvPos')


        self.scp = self.__set_pvp_data( meta, pvp, 'SRPPos')


        self.fx1 = self.__set_pvp_data(meta, pvp, 'FX1')

        self.fx2 = self.__set_pvp_data(meta, pvp, 'FX2')
        self.aFDOP = self.__set_pvp_data( meta, pvp, 'aFDOP')
        self.aFRR1 = self.__set_pvp_data( meta, pvp, 'aFRR1')
        self.aFRR2 = self.__set_pvp_data( meta, pvp, 'aFRR2')

        self.signal = self.__set_pvp_data(meta, pvp, 'SIGNAL')

        self.fx0 = self.__set_pvp_data( meta, pvp, 'SC0')

        self.fxss = self.__set_pvp_data(meta, pvp, 'SCSS')


    def __set_pvp_data(self, meta, pvp, search_key):

        ## perform a search on the key


        items = list(meta.PVP.to_dict().items())

        ind = [idx for idx, key in enumerate(items) if key[0] == search_key]

        if ind:

            arr = np.array( [val[ind[0]] for val in pvp ]  )

            return arr

        else:

            print( f'pvp key name,{search_key}, not found' )

            return np.zeros(len(pvp))

