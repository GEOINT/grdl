# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:42:29 2024

@author: User
"""

import numpy as np

### get all the base SICD types
from sarpy.io.complex.sicd_elements.Antenna import AntennaType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType
from sarpy.io.complex.sicd_elements.ErrorStatistics import ErrorStatisticsType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType

from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType, TxFrequencyProcType,ProcessingType
from sarpy.io.complex.sicd_elements.MatchInfo import MatchInfoType
from sarpy.io.complex.sicd_elements.PFA import PFAType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.RMA import RMAType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType
from sarpy.io.complex.sicd_elements.RgAzComp import RgAzCompType
from sarpy.io.complex.sicd_elements.SCPCOA import SCPCOAType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd import SICDWriter

class Make_SICD():
    
    def __init__(self, meta, geometry, polargrid):
        '''
        Build the object that is passed around that builds the meta data for a 
        SICD and writes it
        '''
        self.meta = meta
        self.geo = geometry
        self.pg = polargrid


        
        #by default this sets it to none
        self.sicd_meta = SICDType()  
        
    def populate_fields(self):
        '''
        helper function call to populate all the meta data fields
        '''
        self.populate_CollectionInfo()
        self.popluate_GeoData()
        self.populate_ImageData()
        self.populate_Timeline()
        self.populate_Grid()
        self.populate_Position()
        self.populate_SCPCOA()
        self.populate_ImageFormation()
        self.populate_ImageForm
        
    def populate_CollectionInfo(self):
        '''
        populate the CollectionInfo metadata
        '''
        collectorname = self.meta.cphdmeta.CollectionID.CollectorName
        corename = self.meta.cphdmeta.CollectionID.CoreName
        radarmode = self.meta.cphdmeta.CollectionID.RadarMode
        classification =self.meta.cphdmeta.CollectionID.Classification
        colinfo = CollectionInfoType(collectorname,
                                     None,
                                     corename,
                                     None,
                                     radarmode,
                                     classification,
                                     None,
                                     None)
        self.sicd_meta.CollectionInfo = colinfo
        
    def popluate_GeoData(self):
        '''
        populate the GeoData field:
            fields: 'EarthModel', 'SCP', 'ImageCorners', 'ValidData'
            required: 'EarthModel', 'SCP', 'ImageCorners'
        '''
        
        EarthModel = 'WGS_84'
        scp_ecf = self.geo.SRP[0]
        scp_llh = self.geo.SRP_LLH[0]
        scp_llh[0] *= 180/np.pi
        scp_llh[1] *= 180/np.pi
        scp = SCPType(scp_ecf, scp_llh)
        imagecorners = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        gd = GeoDataType(EarthModel,
                         scp,
                         imagecorners)
        self.sicd_meta.GeoData = gd
        
        
    def populate_ImageData(self):
        ### build the ImageDataType
        from sarpy.io.complex.sicd_elements.ImageData import ImageDataType, FullImageType
        pixeltype = 'RE32F_IM32F'
        NumRows = self.pg.recNSamples     
        NumCols = self.pg.recNPulses
        FirstRow = 0
        FirstCol = 0
        scppixel = [NumRows//2, NumCols//2]
        validdata = [[0,0], [0, NumCols-1], [NumRows-1, NumCols-1], [NumRows-1,0]]
        idt = ImageDataType(pixeltype,
                      None,
                      NumRows,
                      NumCols,
                      FirstRow,
                      FirstCol,
                      FullImageType(NumRows,NumCols),
                      scppixel,
                      validdata)
        self.sicd_meta.ImageData = idt
    
    def populate_Timeline(self):
        '''
        populate the Timeline meta data field:
            fields:'CollectStart', 'CollectDuration', 'IPP'
            Required:'CollectStart', 'CollectDuration'

        Returns
        -------
        None.

        '''
        collectstart = self.meta.cphdmeta.Global.Timeline.CollectionStart
        collectdur = self.meta.time[-1]-self.meta.time[0]
        
        tl = TimelineType( collectstart,
                           collectdur,
                           None)
        self.sicd_meta.Timeline = tl
    
    def populate_Grid(self):
        '''
        populate the Grid metadata
            fields:'ImagePlane', 'Type', 'TimeCOAPoly', 'Row', 'Col'
            required: 'ImagePlane', 'Type', 'TimeCOAPoly', 'Row', 'Col'

        Returns
        -------
        None.

        '''
        imageplane = self.geo.imageplane
        planeType = 'RGAZIM'
        sgn = -1
        deltakcoaPoly = [[0]]
        ### row vector, range info
        uvectecf =self.geo.rg_uvectecf
        ss = self.pg.rg_ss
        impresbw = self.pg.rg_impresbw
        impreswid = self.pg.rg_impreswid
        kctr = self.pg.rg_kctr
        deltak1 = self.pg.rg_deltak1
        deltak2 = self.pg.rg_deltak2
        row = DirParamType(uvectecf,
                           ss,
                           impreswid,
                           sgn,
                           impresbw,
                           kctr,
                           deltak1,
                           deltak2,
                           deltakcoaPoly,
                           None,
                           None)
        uvectecf =self.geo.az_uvectecf
        ss = self.pg.az_ss
        impresbw = self.pg.az_impresbw
        impreswid = self.pg.az_impreswid
        kctr = self.pg.az_kctr
        deltak1 = self.pg.ku_bounds[0]
        deltak2 = self.pg.ku_bounds[1]
        col = DirParamType(uvectecf,
                           ss,
                           impreswid,
                           sgn,
                           impresbw,
                           kctr,
                           deltak1,
                           deltak2,
                           deltakcoaPoly,
                           None,
                           None)
        
        gt = GridType(imageplane,
                      planeType,
                      None,
                      row,
                      col)
        self.sicd_meta.Grid = gt
        a = 1
    
    def populate_Position(self):
        '''
        populate the position poly
            field: ARPPoly', 'GRPPoly', 'TxAPCPoly', 'RcvAPC'
            required: 'ARPPoly'

        Returns
        -------
        None.

        '''
        
        arp_poly = XYZPolyType(self.geo.arp_poly_x.coef,
                    self.geo.arp_poly_y.coef,
                    self.geo.arp_poly_z.coef)
        
        pos = PositionType(arp_poly,
                           None,
                           None,
                           None)
        self.sicd_meta.Position = pos
        
        a = 1
    
    def populate_SCPCOA(self):
        '''
        populate the SCPCOA, scene center point center of aperture type
            fields:'SCPTime', 'ARPPos', 'ARPVel', 'ARPAcc', 'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAng',
            'GrazeAng', 'IncidenceAng', 'TwistAng', 'SlopeAng', 'AzimAng', 'LayoverAng'
            required: 'SCPTime', 'ARPPos', 'ARPVel', 'ARPAcc', 'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAng',
            'GrazeAng', 'IncidenceAng', 'TwistAng', 'SlopeAng', 'AzimAng', 'LayoverAng'

        Returns
        -------
        None.

        '''
        arp_coa = [self.geo.arp_poly_x(self.geo.COATime),
                     self.geo.arp_poly_y(self.geo.COATime),
                     self.geo.arp_poly_z(self.geo.COATime)]
        arp_vel_coa = [self.geo.arp_poly_x.deriv(1)(self.geo.COATime),
                     self.geo.arp_poly_y.deriv(1)(self.geo.COATime),
                     self.geo.arp_poly_z.deriv(1)(self.geo.COATime)]
        arp_acc_coa = [self.geo.arp_poly_x.deriv(2)(self.geo.COATime),
                     self.geo.arp_poly_y.deriv(2)(self.geo.COATime),
                     self.geo.arp_poly_z.deriv(2)(self.geo.COATime)]
        scp =SCPCOAType(self.geo.COATime,
                        arp_coa,
                        arp_vel_coa,
                        arp_acc_coa,
                        self.geo.SideOfTrack,
                        self.geo.SlantRange_COA,
                        self.geo.GroundRange_COA,
                        self.geo.DopConeAng_COA*180/np.pi,
                        self.geo.GrazAng_COA*180/np.pi,
                        self.geo.IncdAng_COA*180/np.pi,
                        self.geo.TwistAng_COA*180/np.pi,
                        self.geo.SlopeAng_COA*180/np.pi,
                        self.geo.AzimAng_COA*180/np.pi,
                        self.geo.LayoverAng_COA*180/np.pi)
        
        self.sicd_meta.SCPCOA = scp
    
    def populate_ImageFormation(self):
        '''
        
            fields: 'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc',
            'TxFrequencyProc', 'SegmentIdentifier', 'ImageFormAlgo', 'STBeamComp',
            'ImageBeamComp', 'AzAutofocus', 'RgAutofocus', 'Processings',
            'PolarizationCalibration'
            required: 'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc', 'TxFrequencyProc',
            'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus'

        Returns
        -------
        None.

        '''
        numchansproc = 1
        prfscalefactor = None
        chanindices = [1]
        rcvchanproc = RcvChanProcType(numchansproc,
                                      prfscalefactor,
                                      chanindices)
        
        
        txrcvpol = self.meta.cphdmeta.TxRcv.RcvParameters[0].Polarization
        txrcvpol = 'OTHER'
        tstart = self.geo.tstart
        tend = self.geo.tend
        
        procFreq = self.pg.kv_bounds/self.pg.sf_conv        
        algo = 'PFA'
        STBeamcomp = 'NO'
        imagebeamcomp = 'NO'
        AzautoFoc = 'NO'
        RgautoFoc = 'NO'
        
        ptype = self.pg.grid_form
        params = {'Processor':'Smalley Mega Super Fun Happy processor not to be used by any serious person, but use it'}
        ptype = ProcessingType(ptype,
                               False,
                               params)
        
        imform = ImageFormationType(rcvchanproc,
                                    txrcvpol,
                                    tstart,
                                    tend,
                                    procFreq,
                                    None,
                                    algo,
                                    STBeamcomp,
                                    imagebeamcomp,
                                    AzautoFoc,
                                    RgautoFoc,
                                    ptype,
                                    None)
        
        self.sicd_meta.ImageFormation = imform
        
    def populate_ImageForm(self):
        self.sicd_meta.ImageForm = 'PFA'        
        
    def write_sicd(self, data, outfilename):
        writer = SICDWriter(outfilename,self.sicd_meta, check_existence=False)
        writer.write( data )
        a = 1