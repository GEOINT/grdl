# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:31:59 2024

@author: User
"""



import numpy as np

from numpy.linalg import norm

from mpl_toolkits.basemap import Basemap


from sarpy.geometry.geocoords import wgs_84_norm,ecf_to_geodetic,ecf_to_enu

class Get_Coordinates:

    def __init__(self, meta):
        '''
        

        Parameters
        ----------
        meta : Meta_Setup
            The per vector parameters in the data. This is the pulse to pulse 
            position and time information for the antenna phase center

        Returns
        -------
        None.

        '''
        self.meta = meta
        self.build_reference_vectors()
        self.build_reference_geometry()
        a = 1
    
    def build_reference_vectors(self):
        '''
        the geometry vectors are built for the monostatic case following 
        the NGA standard as defined in the CPHD documentation
        NGA.STND.0068-1_1.1.0, section 6.5.1
        '''
        
        ### get the scene center point ecf
        self.SRP = self.meta.scp.copy()
        ## get the scene center point in wgs-84, returns it in degrees
        self.SRP_LLH = ecf_to_geodetic(self.SRP, ordering= 'latlong')
        ###scene reference point magnitude
        self.SRP_DEC = norm( self.SRP, axis = 1).reshape( -1,1)
        ##unit vector in earth center coordinates for the scp
        self.uEC_SCP = 1/self.SRP_DEC * self.SRP
        
        srp = self.SRP_LLH[0].copy()
        srp[0] *= np.pi/180
        srp[1] *= np.pi/180
        
        ### convert the SRP to ENU 
        self.uEAST = np.array( [-np.sin( srp[1]),
                                np.cos( srp[1]),
                                0])
        
        self.uNOR = np.array( [-np.sin(srp[0])*np.cos( srp[1]),
                               -np.sin(srp[0])*np.sin( srp[1]),
                               np.cos( srp[0])])
        
        self.uUP = np.array( [np.cos(srp[0])*np.cos( srp[1]),
                              np.cos(srp[0])*np.sin( srp[1]),
                              np.sin( srp[0])])
        a = 1
        
    def build_reference_geometry(self):
        '''
        the geometry vectors are built for the monostatic case following 
        the NGA standard as defined in the CPHD documentation
        NGA.STND.0068-1_1.1.0, section 6.5.1
        '''
        
        
        self.txrange = self.meta.txpos - self.meta.scp
        self.rcvrange = self.meta.rcvpos - self.meta.scp
        
        self.txmag = norm( self.txrange, axis =1).reshape(-1,1)
        self.rcvmag = norm( self.rcvrange, axis = 1).reshape(-1,1)
        
        self.txunit = self.txrange / self.txmag
        self.rcvunit = self.rcvrange / self.rcvmag
        
        bisector = self.txunit + self.rcvunit
        bisector = bisector / norm( bisector, axis = 1).reshape( -1,1)
        
        self.range_vec = bisector*(self.txmag + self.rcvmag)/2
        
        ### I changed the uARP to uRangeVec, in the NGA manual ARP refers to ECF while
        ### uARP refers to the range vector
        uRangeVec = self.range_vec / norm( self.range_vec, axis = 1 ).reshape(-1,1)
        
        ### start by building the Antenna reference point vector
        ### this is with respect to ECF
        self.ARP = np.mean( [self.meta.txpos, self.meta.rcvpos], axis =0)
        self.VARP = np.mean( [self.meta.txvel, self.meta.rcvvel],axis = 0)        
        
        ### now compute the ARP with respect to the SRP
        ##magnitude of the / range of the ARP - scp
        R_ARP_SRP = norm( self.ARP - self.meta.scp, axis = 1 ).reshape(-1,1)
        ##unit vector
        uARP = 1/R_ARP_SRP*(self.ARP - self.meta.scp)
        
        ### next get the range rate in meters per second
        Rdot_ARP_SRP = np.sum( uARP*self.VARP, axis = 1)
    
        ### get the magnitude of the antenna reference point win ECF
        ARP_DEC = norm( self.ARP, axis = 1).reshape(-1,1)
        
        ##compute the unit normal vector to the earth center for the ARP
        uEC_ARP = self.ARP/ARP_DEC
        
        ##earth angle, antenna reference point
        EA_ARP = np.arccos(np.sum(self.uEC_SCP * uEC_ARP,axis = 1) )
        ## computer the ground range from the antenna reference point
        ## to the scene reference point
        Rg_ARP_SCP = np.hstack(self.SRP_DEC)*EA_ARP
        
        ### get the magnitude of the velocity antenna reference point
        VARP_M = norm( self.VARP, axis = 1 ).reshape(-1,1)
        
        ### get the unit vector of the velocity antenna reference point
        uVARP = 1/VARP_M*self.VARP
        
        ### points to the left side of the ARP ground track
        LEFT = np.cross( uEC_ARP, uVARP )
        ###LOOK = +1 for a left looking collection, -1 for a right looking collection
        LOOK = -1
        Side_Of_Track = 'R'
        
        left_uarp_dot = np.sum( LEFT*uARP, axis = 1)
        
        if left_uarp_dot[0] < 0:
            LOOK = 1
            Side_Of_Track = 'L'
        
        ###next get the doppler cone angle
        DCA = np.arccos( -Rdot_ARP_SRP / np.hstack(VARP_M) )
        
        ### define the ground plane
        uGPZ = wgs_84_norm( self.SRP )
        uGPY = np.cross( uGPZ, uRangeVec)
        uGPY = uGPY / norm( uGPY, axis  =1 ).reshape(-1,1)
        uGPX = np.cross( uGPY, uGPZ)
        
        ### graze angle as a function of pulse
        GRAZ = np.arccos( np.sum( uRangeVec * uGPX, axis =1 ))
        INCD = np.pi/2 - GRAZ
        
        GPX_N = np.dot( uGPX, self.uNOR)
        GPX_E = np.dot( uGPX, self.uEAST)
        ### azim in file = 325.2307619027581
        AZIM = np.arctan2( GPX_E, GPX_N)
    
        SPN = LOOK*np.cross( uARP, uVARP)
        uSPN = SPN / norm( SPN, axis = 1).reshape(-1,1)
        
        TWST = -np.arcsin( np.sum( uSPN* uGPY, axis = 1))
        
        SLOPE = np.arccos( np.sum( uGPZ*uSPN,axis = 1)) 
        
        ### layover direction, the direction that elevated targets will be displaced
        ## in ground plane image, LO in file 282.16161354945365
        LODIR_N = -np.dot( uSPN, self.uNOR)
        LODIR_E = -np.dot( uSPN, self.uEAST)
        
        ### the look angle
        LO_ANG = np.arctan2( LODIR_E, LODIR_N)
        
        self.LayoverAng = LO_ANG
        self.SlopeAng = SLOPE
        self.TwistAng = TWST
        self.AzimAng = AZIM
        self.IncdAng = INCD
        self.GrazAng = GRAZ
        self.DopConeAng = DCA
        self.SideOfTrack = Side_Of_Track
        self.Rg_ARP_SCP = Rg_ARP_SCP
        a = 1

    def projection( self, points, line_direction, point_in_plane, plane_normal):
        '''
        standard call is:
        points: points to project
        line_direction: focal plane normal (fpn)
        point_in_plane: scene center point (scp)
        plane_normal: image plane normal (ipn)
        '''
        plane_normal = np.hstack( plane_normal)
        
        dot = np.sum( line_direction*plane_normal,axis = 1)
        dot = np.dot( line_direction,plane_normal )
        d = np.dot(point_in_plane - points, plane_normal) / dot
        return points + (np.vstack(d)*np.vstack( line_direction ))

    def get_geo(self, pv, scp, pv_coa, ipn, fpn):
        ip_pos = self.projection(pv, fpn, scp, ipn)
        ip_coa_pos = self.projection( pv_coa, fpn, scp, ipn)
        
        ipx = ip_coa_pos - scp 
        ipx = ipx / norm( ipx )
        ipy = np.cross( ipx, ipn)
        
        ip_range_vectors = ip_pos - scp
        self.phi = -np.arctan2( np.sum( ip_range_vectors* ipy, axis = 1 ), np.sum( ip_range_vectors*ipx, axis = 1  ) )
        
        range_vectors = pv - scp
        range_vectors = range_vectors / norm( range_vectors, axis = 1).reshape(-1,1)
        sin_graze  = np.sum( range_vectors*fpn, axis = 1 )
        ip_range_vectors = ip_range_vectors / norm( ip_range_vectors, axis = 1).reshape(-1,1)
        sin_graze_ip = np.sum( ip_range_vectors*fpn, axis = 1)
        self.k_sf = np.sqrt(1-(sin_graze**2)) / np.sqrt( 1 - (sin_graze_ip**2))
        self.vpmag = 1
        self.Theta = np.abs(self.phi[-1] - self.phi[0])
        self.az_uvectecf = ipy[0].copy()
        self.rg_uvectecf = ipx[0].copy()
        
        self.arp_poly_x = np.polynomial.Polynomial.fit( self.meta.time, self.ARP[:,0], 5)
        self.arp_poly_y = np.polynomial.Polynomial.fit( self.meta.time, self.ARP[:,1], 5)
        self.arp_poly_z = np.polynomial.Polynomial.fit( self.meta.time, self.ARP[:,2], 5)

        a =1

    # def define_focus_plane( self ):
    #     ### the focus plane is a fundamental geometric plane of the collection
    #     ### this will be defined as Eichel from SNL defines it
    #     ### with a bar indicating the focus plane
    #     zbar = wgs_84_norm( self.meta.scp[0] )
        
    #     p_vec = self.range_vec
        
    #     vhat_f1 = (p_vec[0,:] - np.dot(p_vec[0,:], zbar)*zbar)/ norm(p_vec[0,:] - np.dot(p_vec[0,:], zbar)*zbar)
    #     vhat_fN = (p_vec[-1,:] - np.dot(p_vec[-1,:], zbar)*zbar)/ norm(p_vec[-1,:] - np.dot(p_vec[-1,:], zbar)*zbar)

    #     self.vhat_f0 = vhat_f1
    #     self.vhat_fn = vhat_fN

    #     ybar = (vhat_f1 + vhat_fN) / norm(vhat_f1 + vhat_fN)
    #     xbar = np.cross( ybar, zbar )
        
    #     self.xbar = xbar 
    #     self.ybar = ybar
    #     self.zbar = zbar
        
    # def define_slant_plane(self):
    #     theta0 = np.arccos( np.dot( self.vhat_f0, self.xbar) )
    #     thetaN = np.arccos( np.dot( self.vhat_fn, self.xbar) )
        
    #     v_fi = (self.range_vec - np.vstack(np.dot( self.range_vec,self.zbar ) )*self.zbar)
    #     vhat_fi = v_fi / norm( v_fi, axis = 1).reshape(-1,1)

    #     theta_i = np.arccos( np.dot(vhat_fi, self.xbar) )
        
    #     theta_span = theta_i[-1] - theta_i[0]
        
    #     ind_theta_third = np.max( np.where( theta_i < (theta_i[0]+ theta_span/3) ) )
    #     ind_theta_twothird = np.max( np.where( theta_i < (theta_i[0]+ 2*theta_span/3) ) )

    #     theta_half = np.mean( theta_i )
    #     ihalf = np.min( np.where( theta_i > theta_half ))
        
    #     zhat = np.cross( self.range_vec[ind_theta_third], self.range_vec[ind_theta_twothird]) 
    #     zhat = zhat / norm( zhat )
        
    #     if np.dot( zhat, self.zbar ) < 0:
    #         zhat *= -1
        
    #     r_los = 1/2*(self.range_vec[ind_theta_third] + self.range_vec[ind_theta_twothird])
    #     yhat = r_los / norm(r_los)
    #     xhat = np.cross( yhat, zhat)
        
    #     self.yhat = yhat
    #     self.xhat = xhat
    #     self.zhat = zhat
    #     self.r_los = r_los
        
        
    #     a = 1

    # def define_image_plane(self):
    #     self.ztild = self.zhat.copy()
        
    #     project = np.inner( self.r_los, self.ztild ) / np.inner( self.ztild, self.zbar ) *self.zbar
    #     ytild = self.r_los - project / norm( self.r_los - project )
    #     xtild = np.cross( ytild, self.ztild)
        
    #     self.ytild = ytild
    #     self.xtild = xtild
    #     self.azimuth = np.arctan2(self.xtild[2] , self.ytild[2] )
        
    #     proj = np.vstack(np.dot( self.range_vec, self.ztild ) / np.inner( self.ztild, self.zbar ) )*self.zbar
    #     p_prime = self.range_vec - proj
    #     self.proscal = norm(self.range_vec, axis = 1) / norm( p_prime, axis = 1 )
        
    #     projlos = np.dot( self.r_los, self.ztild ) / np.dot( self.ztild, self.zbar )*self.zbar

    #     r_prime_los = self.r_los - projlos
        
        
    #     self.vpmag = norm( r_prime_los ) / norm( self.r_los) 
    #     self.phi = -np.arctan2( np.dot(self.range_vec, self.xtild), np.dot(self.range_vec, self.ytild))
    #     self.Theta = np.abs(self.phi[-1] - self.phi[0])
    
    #     self.poly_fit = np.polynomial.Polynomial.fit(self.meta.time, self.phi,5,)
        
    
    def get_coordinates(self, slant = True):
        #self.define_focus_plane()
        #self.define_slant_plane()
        #self.define_image_plane()

        ### define the coordinate system everything sits in ECEF at first
        center_Index = self.meta.npulses//2
        coaTime = self.meta.time[center_Index]
        self.COATime = coaTime
        #range_vec_vel_coa = np.diff( self.range_vec[center_Index + [1, -1]], axis = 0)
        range_vec_vel_coa = self.VARP[center_Index] -self.meta.scp
        self.COAARPVel = range_vec_vel_coa
        #image scene reference point. Since everything is now with respect to SRP 
        # this is set to 0,0,0
        look_point = [0,0,0]
        ###focal plane norml
        fpn = wgs_84_norm(self.meta.scp)
        ### this is the line of sight vector
        self.range_vec_COA = self.range_vec[center_Index]
        
        ## this projects the data to be north up
        #self.range_vec_COA = np.dot( self.range_vec_COA, self.uNOR) / norm( self.uNOR) * self.uNOR
        
        self.SlantRange_COA = norm( self.range_vec_COA)
        #slant range vector 
        self.srv = self.range_vec_COA 
        
        #by default set the image plane normal to slant plane
        ipn = np.cross( self.range_vec[0], self.range_vec[-1])
        self.ipn = ipn/norm( ipn)
        self.imageplane = 'SLANT'
        if slant == False:
            self.imageplane = 'GROUND'
            ### set the image plane normal to the focal plane normal, aka the ground plane
            self.ipn = fpn
            ### set the image plane normal to the up direction in ENU
            self.ipn = self.uUP
        
        self.get_geo(self.range_vec, look_point, self.range_vec_COA, self.ipn, fpn)


        self.LayoverAng_COA = self.LayoverAng[center_Index]
        self.SlopeAng_COA = self.SlopeAng[center_Index]
        self.TwistAng_COA = self.TwistAng[center_Index]
        self.AzimAng_COA = self.AzimAng[center_Index]
        self.IncdAng_COA = self.IncdAng[center_Index]
        self.GrazAng_COA = self.GrazAng[center_Index]
        self.DopConeAng_COA = self.DopConeAng[center_Index]
        self.GroundRange_COA = self.Rg_ARP_SCP[center_Index]
        
        self.tstart = self.meta.time[0]
        self.tend = self.meta.time[-1]
        
        #self.get_geo(self.range_vec, look_point, self.range_vec_COA, self.ipn, np.hstack( self.ipn ) )

        ###set zbar in this case the coordinate center is the scp

        zbar = wgs_84_norm(self.meta.scp[0])
        #zbar = self.meta.scp[0] / norm( self.meta.scp[0])
        #zbar = np.array( [0.11369559667581591, -0.8269010128992413, 0.5507340793547633] )
        
        print( 'Focal Plane Normal FPN: ', zbar)

        npulses = self.meta.npulses
        nsamples = self.meta.nsamples
        self.npulses = npulses
        self.nsamples = nsamples
        c = 299792458.0
        self.c = c
        B = self.meta.fx1[0] - self.meta.fx2[0]
        self.B = B
        fad = B/nsamples
        self.fad = np.mean( self.meta.fxss )  ###
        fst = self.meta.fx1


    def plot_collection_geometry(self):
        import matplotlib.pyplot as plt
        fig  = plt.figure()
        
        ### set a 250 km x 250 km scp view
        bm = Basemap(projection = 'lcc', 
                      resolution = 'l', 
                      width = 2.5e6,
                      height = 2.5e6,
                      lat_0=int(self.SRP_LLH[0,0]), 
                      lon_0=int(self.SRP_LLH[0,1]))
        # bm.drawmapboundary(fill_color='aqua')
        # bm.fillcontinents(color='coral',lake_color='aqua')
        # bm.drawcoastlines()
        bm.bluemarble(scale = 0.5)
        #bm.etopo(scale= 1.5)
        
        bm.drawcountries()
        bm.drawstates()
        #yticks = np.linspace( bm.latmin, bm.latmax, 10)
        #ax.set_yticks( yticks ) 
        ax = fig.gca()

        px,py = bm( self.SRP_LLH[0,1],self.SRP_LLH[0,0])
        bm.plot( px, py, 'bo')
        plt.title('Collection scene center point')
        

        
        a=1
