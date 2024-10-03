# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:04:24 2024

@author: User
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt

def RMA_Algo(meta, geo ):
    
    ##pulse compress the data:
    dat = meta.reader[None,None]    
    dat = np.fft.fftshift( np.fft.fft( dat, axis = 0), axes = 0)
    freq =  np.arange( meta.nsamples )*np.mean(meta.fxss) + np.mean( meta.fx0)
    #freq = np.reshape( freq, (1, meta.nsamples))
    krange = 2*freq/3e8
    krange = np.reshape( krange, (1, meta.nsamples))
    #prf = 2*np.pi/ np.mean( np.diff( meta.txtime ) )
    dis = np.linalg.norm( meta.rcvpos - meta.txpos, axis = 1 )
    m_Dis = np.mean(dis)
    kaz = np.linspace( -np.pi / m_Dis, np.pi /m_Dis , meta.npulses ) 
    kaz = np.reshape( kaz, (meta.npulses, 1))
    kx = krange**2- kaz**2
    kx = np.sqrt( kx)
    
    #krr, kxx = np.meshgrid( krange, kaz)
    #kx = np.sqrt( krr**2 - kxx**2)
    
    plt.figure()
    plt.imshow( (kx ), cmap = 'gray', aspect='auto')
    plt.title('kx')
    
    Rc = np.vstack( geo.Rg_ARP_SCP )
    
    kFinal = np.exp( 1j*kx*Rc)
    
    s_mf = dat*kFinal
    plt.figure()
    plt.imshow( np.abs(s_mf ), cmap = 'gray', aspect='auto')
    plt.title('s_mf')
    stolt = s_mf.copy()
    for i in range( meta.npulses ):
        interp_fn = scipy.interpolate.interp1d(kx[i], s_mf[i], bounds_error=False, fill_value=0)
        stolt[i] = interp_fn(np.hstack( krange))
        #stolt[:,i] = np.interp(krange[0,i], kx[:,i], sdata[:,i] )
    #stolt *=np.exp( -1j*krange*Rc)
    
    plt.figure()
    plt.imshow( np.abs(dat ), cmap = 'gray', aspect='auto', vmax = 1e4)
    plt.title('raw data')
    plt.figure()
    plt.imshow( np.abs(stolt ), cmap = 'gray', aspect='auto')
    plt.title('stolt')
    stolt_fft = np.fft.ifft2( stolt )
    
    plt.figure()
    plt.imshow( np.abs(stolt_fft ), cmap = 'gray', aspect='auto')
    plt.title('stolt fft')
    
    a = 1
    
  #   def RMA(sif, pulse_period=20e-3, freq_range=None, Rs=9.0):
  # '''Performs the Range Migration Algorithm.
  # Returns a dictionary containing the finished S_image matrix
  # and some other intermediary values needed for drawing the image.

  # sif is a NxM array where N is the number of SAR frames and M
  # is the number of samples within each measurement over the time period
  # of frequency modulation increase.

  # freq_range should be a tuple of your starting frequency in a range sample and your final frequency.
  # If given none, the values from MIT will be used. Please consult your VCO's datasheet data otherwise
  # and adjust the constant at the top of this file.

  # Rs is distance (in METERS for just this function) to scene center. Default is ~30ft.
  # '''
  # if freq_range is None:
  #   freq_range = [2260e6, 2590e6] # Values from MIT

  # N, M = len(sif), len(sif[0])

  # # construct Kr axis
  # delta_x = feet2meters(2/12.0) # Assuming 2 inch antenna spacing between frames.
  # bandwidth = freq_range[1] - freq_range[0]
  # center_freq = bandwidth/2 + freq_range[0]
  # Kr = numpy.linspace(((4*PI/C)*(center_freq - bandwidth/2)), ((4*PI/C)*(center_freq + bandwidth/2)), M)

  # # smooth data with hanning window
  # sif *= numpy.hanning(M)

  # '''STEP 1: Cross-range FFT, turns S(x_n, w(t)) into S(Kx, Kr)'''
  # # Add padding if we have less than this number of crossrange samples:
  # # (requires numpy 1.7 or above)
  # rows = (max(2048, len(sif)) - len(sif)) / 2
  # try:
  #   sif_padded = numpy.pad(sif, [[rows, rows], [0, 0]], 'constant', constant_values=0)
  # except Exception, e:
  #   print "You need to be using numpy 1.7 or higher because of the numpy.pad() function."
  #   print "If this is a problem, you can try to implement padding yourself. Check the"
  #   print "README for where to find cansar.py which may help you."
  #   raise e
  # # N may have changed now.
  # N = len(sif_padded)

  # # construct Kx axis
  # Kx = numpy.linspace(-PI/delta_x, PI/delta_x, N)

  # freqs = numpy.fft.fft(sif_padded, axis=0) # note fft is along cross-range!
  # S = numpy.fft.fftshift(freqs, axes=(0,)) # shifts 0-freq components to center of spectrum

  # '''
  # STEP 2: Matched filter
  # The overlapping range samples provide a curved, parabolic view of an object in the scene. This
  # geometry is captured by S(Kx, Kr). Given a range center Rs, the matched filter perfectly
  # corrects the range curvature of objects at Rs, partially other objects (under-compsensating
  # those close to the range center and overcompensating those far away).
  # '''

  # Krr, Kxx = numpy.meshgrid(Kr, Kx)
  # phi_mf = Rs * numpy.sqrt(Krr**2 - Kxx**2)
  # # Remark: it seems that eq 10.8 is actually phi_mf(Kx, Kr) = -Rs*Kr + Rs*sqrt(Kr^2 - Kx^2)
  # # Thus the MIT code appears wrong. To conform to the text, uncomment the following line:
  # #phi_mf -= Rs * Krr
  # # However it is left commented by default because all it seems to do is shift everything downrange
  # # closer to the radar by Rs with no noticeable improvement in picture quality. If you do
  # # uncomment it, consider just subtracting Krr instead of Krr multiplied with Rs.
  # S_mf = S * numpy.exp(1j*phi_mf)

  # '''
  # STEP 3: Stolt interpolation
  # Compensates range curvature of all other scatterers by warping the signal data.
  # '''

  # kstart, kstop = 73, 108.5 # match MIT's matlab -- why are these values chosen?
  # Ky_even = numpy.linspace(kstart, kstop, 1024)

  # Ky = numpy.sqrt(Krr**2 - Kxx**2) # same as phi_mf but without the Rs factor.
  # try:
  #   S_st = numpy.zeros((len(Ky), len(Ky_even)), dtype=numpy.complex128)
  # except:
  #   S_st = numpy.zeros((len(Ky), len(Ky_even)), dtype=numpy.complex)
  # # if we implement an interpolation-free method of stolt interpolation,
  # # we can get rid of this for loop...
  # for i in xrange(len(Ky)):
  #   interp_fn = scipy.interpolate.interp1d(Ky[i], S_mf[i], bounds_error=False, fill_value=0)
  #   S_st[i] = interp_fn(Ky_even)

  # # Apply hanning window again with 1+
  # window = 1.0 + numpy.hanning(len(Ky_even))
  # S_st *= window

  # '''
  # STEP 4: Inverse FFT, construct image
  # '''

  # ifft_len = [len(S_st), len(S_st[0])] # if memory allows, multiply both
  # # elements by 4 for perhaps a somewhat better image. Probably only viable on 64-bit Pythons.
  # S_img = numpy.fliplr(numpy.rot90(numpy.fft.ifft2(S_st, ifft_len)))

  # return {'Py_S_image': S_img, 'S_st_shape': S_st.shape, 'Ky_len': len(Ky), 'delta_x': delta_x, 'kstart': kstart, 'kstop': kstop}
