"""
    Class to read STO-2 Level 0.6 and Level 1.0 data
    
    There can still be bugs in this software. No software is really bug-free.....
    The use of the software is at you own risk.
    
    Created: 1/1/2017
    By:      V. Tolls, SAO
    Current Version: 3.5 (current version see also variable after class definition)
    
    Modification History:
    3/16/2017: modified resample: return input arrays if step=1
    01/10/2018: added masks to each line to log spike removal/repairs and other modifications
    11/02/2018: started version 3.3, primarily to add velocity cropping for [NII] spectra
    03/18/2019: updated the coordinate retrieval to Galactic coordinates
    05/01/2020: v3.5: minor updates to improve functionality
    
"""
import os
import numpy as np
import glob
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import signal
from scipy.signal import butter, filtfilt, medfilt
from scipy import interpolate
from scipy.interpolate import interp1d, CubicSpline
from scipy.interpolate import UnivariateSpline
from pylab import *
import matplotlib.mlab as mlab
import warnings
from ALSFitter import *
from _cffi_backend import __version__

__version__ = '3.5.0'
__date__ = '05/01/2020'
 

warnings.filterwarnings("ignore")
 
# convenience routine to load data from file
def readSTO2(ifile, retdd=False, retcl=False, verbose=False, trim=None, rclip=None, tnorm=True, badpix=None, badpix0=None, badpix1=None, badpix2=None):
    """
    Convenience routine for class rsto2 to read STO-2 Level 0.5 data of a single line
    Input:
        ifile:      data file name (full path might be required)
        retdd:      if True, raw data table dd1 is returned (default: False)
        retcl:      if True, class object is returned (default: False)
        verbose:    print some info to screen
        trim:       trim both ends of the velocity and intensity arrays by the given number of 
                    pixels (default: None) Note: better use rclip to not change array size (shape)!
        rclip:      replaces the given number of pixels at both ends of the velocity and intensity arrays by the value 
                    of the next adjoining pixels (default: None)
        tnorm:      flag to return spectrum divided by integration time: spectrum/tint
        badpix:     pixels with bad values to be replaced by interpolated value from neighboring pixels
        badpix0,1,2: pixels with bad values to be replaced by interpolated value from neighboring pixels for lines 0, 1, or 2 only
    Output:
        separate arrays for velocity, and intensity, pos (astropy SkyCoord object), FITS header, and (optional) the raw data table
    """
    rs = rsto2(ifile, verbose=False, trim=None, rclip=rclip, tnorm=tnorm, badpix=badpix, badpix0=badpix0, badpix1=badpix1, badpix2=badpix2)
    vv, spec = rs.getData()
    
    if retcl:
        if retdd: return vv, spec, rs.getPos(), rs.getHeader(), rs.getRawData(), rs
        else: return vv, spec, rs.getPos(), rs.getHeader(), rs
    else:
        if retdd: return vv, spec, rs.getPos(), rs.getHeader(), rs.getRawData()
        else: return vv, spec, rs.getPos(), rs.getHeader()



def readSTO2Line(ifile, lin, retdd=False, retcl=False, verbose=False, trim=None, rclip=None, tnorm=True, badpix=None, badpix0=None, badpix1=None, badpix2=None):
    """
    Convenience routine for class rsto2 to read STO-2 Level 0.5 data of a single line
    Input:
        ifile:      data file name (full path might be required)
        lin:        desired Level 0.5 line: 0: NII_1; 1: NII_2; 2: CII_1; 3: CII_2; 4: OI_FFT; 5: OI_Freq_LK 
                           Level 0.6 line: 0: NII_1; 1: NII_2; 2: CII_2 (other, not observed lines were removed!)
        retdd:      if True, raw data table dd1 is returned (default: False)
        retcl:      if True, class object is returned (default: False)
        verbose:    print some info to screen
        trim:       trim both ends of the velocity and intensity arrays by the given number of 
                    pixels (default: None) Note: better use rclip!
        rclip:      replaces the given number of pixels at both ends of the velocity and intensity arrays by the value 
                    of the next adjoining pixels (default: None)
        tnorm:      flag to return spectrum divided by integration time: spectrum/tint
        badpix:     pixels with bad values to be replaced by interpolated value from neighboring pixels
    Output:
        separate arrays for velocity, and intensity, pos (astropy SkyCoord object), FITS header, and (optional) the raw data table
    """
    rs = rsto2(ifile, verbose=False, trim=None, rclip=rclip, tnorm=tnorm, badpix=badpix, badpix0=badpix0, badpix1=badpix1, badpix2=badpix2)
    vv, spec = rs.getDataLine(lin)
    
    if retcl:
        if retdd: return vv, spec, rs.getPosLine(lin), rs.getHeader(), rs.getRawData(), rs
        else: return vv, spec, rs.getPosLine(lin), rs.getHeader(), rs
    else:
        if retdd: return vv, spec, rs.getPosLine(lin), rs.getHeader(), rs.getRawData()
        else: return vv, spec, rs.getPosLine(lin), rs.getHeader()



def readSTO2LineM(ifile, lin, retdd=False, retcl=False, verbose=False, trim=None, rclip=None, tnorm=True, 
                  badpix=None, badpix0=None, badpix1=None, badpix2=None, vrange=None):
    """
    Convenience routine for class rsto2 to read STO-2 Level 0.5 data of a single line
    Input:
        ifile:      data file name (full path might be required)
        lin:        desired Level 0.5 line: 0: NII_1; 1: NII_2; 2: CII_1; 3: CII_2; 4: OI_FFT; 5: OI_Freq_LK 
                           Level 0.6 line: 0: NII_1; 1: NII_2; 2: CII_2 (other, not observed lines were removed!)
        retdd:      if True, raw data table dd1 is returned (default: False)
        retcl:      if True, class object is returned (default: False)
        verbose:    print some info to screen
        trim:       trim both ends of the velocity and intensity arrays by the given number of 
                    pixels (default: None) Note: better use rclip!
        rclip:      replaces the given number of pixels at both ends of the velocity and intensity arrays by the value 
                    of the next adjoining pixels (default: None)
        tnorm:      flag to return spectrum divided by integration time: spectrum/tint
        badpix:     pixels with bad values to be replaced by interpolated value from neighboring pixels
        vrange:     if provided, will be used to return spectra limited to this range for simplification in data reduction (default: None)
    Output:
        separate arrays for velocity, and intensity, mask, pos (astropy SkyCoord object), FITS header, and (optional) the raw data table
    """
    rs = rsto2(ifile, verbose=False, trim=None, rclip=rclip, tnorm=tnorm, badpix=badpix, badpix0=badpix0, badpix1=badpix1, badpix2=badpix2)
    vv, spec = rs.getDataLine(lin)
    mask = rs.getDataMaskLine(lin)
    if vrange!=None:
        if verbose: print(vv.shape, spec.shape, mask.shape, vrange, vv.value.min(), vv.value.max())
        vsel = np.where((vv.value>=vrange[0])&(vv.value<=vrange[1]))
        vv = np.squeeze(vv[vsel])
        spec = np.squeeze(spec[vsel])
        mask = np.squeeze(mask[vsel])
        if verbose: print(vv.shape, spec.shape, mask.shape, vrange, vv.value.min(), vv.value.max())
    
    if retcl:
        if retdd: return vv, spec, mask, rs.getPosLine(lin), rs.getHeader(), rs.getRawData(), rs
        else: return vv, spec, mask, rs.getPosLine(lin), rs.getHeader(), rs
    else:
        if retdd: return vv, spec, mask, rs.getPosLine(lin), rs.getHeader(), rs.getRawData()
        else: return vv, spec, mask, rs.getPosLine(lin), rs.getHeader()



class rsto2:
    
    
    def __init__(self, ifile, trim=None, rclip=None, verbose=False, tnorm=True, badpix=None, badpix0=None, badpix1=None, badpix2=None):
        '''
        reading STO-2 Level 0 and 1 data
        the difference is that some entries in the data table dd (see retdd=True) are not
        available in the Level 0 data, e.g. Tsys
        
        ifile:       data file name (full path might be required)
        verbose:     printing some info  (default: False)
        trim:        trim both ends of the velocity and intensity arrays by the given number of 
                     pixels (default: None) Note: better use rclip!
        rclip:       replaces the value of given number of pixels at both ends of the intensity array by the value 
                     of the next adjoining pixels (default: None, but typical value is 1 or 2)
        tnorm:       flag to return spectrum divided by integration time: spectrum/tint
        badpix:      pixels with bad values to be replaced by interpolated value from neighboring pixels
                     using the same badpix indices for each of the 3 spectra
        '''
        if rclip==0: rclip=None
        if trim==0: trim = None
        
        if verbose:
            print('STO_v34 Version: ', __version__,' of ', __date__)
        self.version = __version__
        self.date = __date__
        
        with fits.open(ifile) as hl:
            #hl = fits.open(ifile)
            self.hd0 = hl[0].header.copy()
            self.dd0 = hl[0].data
            self.hd1 = hl[1].header.copy()
            self.dd1 = hl[1].data.copy()
            self.cols = hl[1].columns
        #print(type(self.hd1))
        #print(type(self.dd1))
            hl.close()

        
        # get the coordinates, update 3/18/2019
        # the [NII] R.A.s did not include the cos(dec)-term. => added 3/19/2019
        # in future it should be possible to just use the WCS, but currently it produces the wrong position!
        ndec1 = (np.float(self.hd1['UDP_DEC'])+np.float(self.dd1['CDELT3'][0])/3600.)
        ndec2 = (np.float(self.hd1['UDP_DEC'])+np.float(self.dd1['CDELT3'][1])/3600.)
        pos0 = SkyCoord((np.float(self.hd1['UDP_RA'])+np.float(self.dd1['CDELT2'][0])/3600./np.cos(np.radians(ndec1)))*u.deg, ndec1*u.deg, frame='icrs')
        pos1 = SkyCoord((np.float(self.hd1['UDP_RA'])+np.float(self.dd1['CDELT2'][1])/3600./np.cos(np.radians(ndec2)))*u.deg, ndec2*u.deg, frame='icrs')
        pos2 = SkyCoord( np.float(self.hd1['UDP_RA'])*u.deg, np.float(self.hd1['UDP_DEC'])*u.deg, frame='icrs')
        self.pos = [pos0.galactic, pos1.galactic, pos2.galactic]
        
        self.spec = self.dd1['DATA'] * u.count
        # check first if the file has already been processed and the pixel mask was added to the file:
        try: 
            self.mask = self.dd1['MASK']
            #print('mask exists.')
        except: 
            # create a empty mask => but mask needs to be filled with header comment info
            self.mask = np.zeros(self.spec.shape, dtype=np.int32)
            # fill the new mask with the header information
            self.mask = addHeaderBadPixels(self.hd1['Comment'], self.mask)
            #print('added mask.')
        self.n_pix = self.dd1['MAXIS1'][0]
        self.n_row = self.hd1['NAXIS2']
        self.vv = np.ndarray((self.n_row,self.n_pix), dtype=np.float) * u.km / u.s
        for i in range(self.n_row):
            self.n_pixl = self.dd1['MAXIS1'][i]   # number of pixels in spectrum
            self.vv[i,:self.n_pixl] = (np.float(self.hd1['CRVAL1']) + (1 + np.arange(self.n_pixl) - self.dd1['CRPIX1'][i]) * self.dd1['CDELT1'][i]) * u.km / u.s
            
        self.vv/= 1000.0
        self.hd1['TUNIT3'] = 'km/s' 

    
        if tnorm:
            tint = np.float(self.hd1['OBSTIME'])
            self.spec = self.spec/tint / u.s
            self.hd1['TUNIT24'] = 'counts/sec'

            
#         if verbose: 
#             #     # 3076  4014   REF    33    CII_2   2.226  322.99963    1.90015  -50.0   36.8  17.585350 -0.00370 -57.09700  2760264
#             print(' scan obsid  Type ot_row   line  obstime     l          b        vLSR     v0    syn_CII biasvolt biascurr totpower')
#             print('%5s %5s %5s %5s %8s %7.3f %10.5f %10.5f %6.1f %6.1f %10.6f %8.5f %8.3f %8.0f'%(
#                 hd1['SCAN'], hd1['OBSID'], hd1['TYPE'], hd1['OT_ROW'], dd1['TELESCOP'], 
#                 np.float(hd1['OBSTIME']), pos.galactic.l.deg, pos.galactic.b.deg, 
#                 np.float(hd1['VELOCITY'])/1000., vv[0], np.float(hd1['SYN_C_II']),
#                 np.float(dd1['BIASVOLT'][lin]),np.float(dd1['BIASCURR'][lin]),np.float(dd1['TOTPOWER'][lin])))
    
        if trim!=None:
            # cut away the outmost pixels - but better to use rclip below!
            vv = vv[:,trim:-trim]
            spec = spec[:,trim:-trim]
    
        if rclip!=None:
            # replace the values of the outermost pixels [0,rclip-1] and [-rclip,last_pixel] with the value 
            # of the first inside pixel [rclip] and [-rclip-1] respectively
            # to remove edge effects, but keep the number of pixels the same
            if verbose: print('STO2_v1: clipping edge values....', rclip)
            for rcl in range(rclip):
                self.spec[:,rcl]   = self.spec[:,rclip]
                self.spec[:,-rcl-1] = self.spec[:,-rclip]
        
        self.badpix = None
        self.badpix0 = None
        self.badpix1 = None
        self.badpix2 = None
        # the following actions need a logging in the mask array!
        if badpix!=None:
            # replace the pixel with indices in badpix with interpolated values
            # repPix works only on 1-D spectra, call it for each of the 3 spectra
            self.badpix = badpix
            for i in range(3):
                self.spec[i,:] = repPix1D(self.spec[i,:], badpix)
        
        if badpix0!=None:
            # replace the pixel with indices in badpix with interpolated values in line 0
            # repPix works only on 1-D spectra, call it for each of the 3 spectra
            self.badpix0 = badpix0
            self.spec[0,:] = repPix1D(self.spec[0,:], badpix0)
        
        if badpix1!=None:
            # replace the pixel with indices in badpix with interpolated values in line 0
            # repPix works only on 1-D spectra, call it for each of the 3 spectra
            self.badpix1 = badpix1
            self.spec[1,:] = repPix1D(self.spec[1,:], badpix1)
        
        if badpix2!=None:
            # replace the pixel with indices in badpix with interpolated values in line 0
            # repPix works only on 1-D spectra, call it for each of the 3 spectra
            self.badpix2 = badpix2
            self.spec[2,:] = repPix1D(self.spec[2,:], badpix2)
            
        
        

    def getDataAsFitsBlocks(self):
        """
        return the data in the file as FITS block: data block, header block
        """
        return self.dd1, self.hd1
        
        
    def getData0AsFitsBlocks(self):
        """
        return the data in the file as FITS block: data block, header block
        """
        return self.dd0, self.hd0
        
        
    def getDataAsArrays(self):
        """
           return a numpy structured array containing all data
        """
        return np.rec.array([self.vv, self.spec],
                            dtype=[('velo', 'f4', (self.n_row, self.n_pixl)),('spec', 'f4', (self.n_row, self.n_pixl))])

    def getDataLine(self, lin):
        """
            Function to retrieve STO2 spectra for a single line:
            Input:
                lin: line number (0: [NII] pixel 1, 1: [NII] pixel 2, and 2: [CII])
            Output:
                velocity array of line, intensity array of line
        """
        return self.vv[lin,:], self.spec[lin,:]
    
    def getData(self):
        """
            Function to retrieve STO2 spectra for all lines:
            Input:
                none
            Output:
                velocity array, intensity array
        """
        return self.vv, self.spec
    
    def getDataMask(self):
        """
            Function to retrieve STO2 data mask for all lines:
            Input:
                none
            Output:
                32-bit integer mask array
        """
        return self.mask
    
    def getDataMaskLine(self, line):
        """
            Function to retrieve STO2 data mask for all lines:
            Input:
                line index (0: NII line 1, 1: NII line2, or 2: CII)
            Output:
                32-bit integer mask array
        """
        return self.mask[line,:,]
    
    def getEmptyMask(self):
        """
            creates an empty mask to accompany the Line data
            the mask elements are 16-bit unsigned integers
        """
        return np.zeros(np.squeeze(self.spec[0,:]).shape, dtype=np.uint16)
        
    def getPos(self):
        """
            returns array of astropy SkyCoord objects (Galactic coordinates)
        """
        return self.pos
    
    def getPosLine(self, lin):
        """
            returns astropy SkyCoord object (Galactic coordinates)
        """
        return (self.pos)[lin]
    
    def getHeader0(self):
        """
            return full FITS header 0
        """
        return self.hd0
    
    def getHeader(self):
        """
            return full FITS header of observation (1st Extension)
        """
        return self.hd1
    
    def getRawData(self):
        """
            return raw data table for all lines
            Table Keywords: MAXIS1, CRPIX1, CDELT1, CDELT2, CDELT3, BiasVolt, BiasCurr, TotPower, IFPower, 
                            BiasPot, PIDstate, LNABias, LNACurr, BeamFac, TSYS, Trx, OutRange, IntTime,
                            NumOff, Telescop, Line, RESTFREQ, IMAGFREQ, DATA
        """
        return self.dd1
    
    def getRawData0(self):
        """
            return (empty) raw data for primary extension
        """
        return self.dd0
    
    def getDataColumns(self):
        """
            return (empty) raw data for primary extension
        """
        return self.cols
    
    def getTint(self):
        """
            return the integration time for this observation
        """
        return np.array(self.hd1['OBSTIME'], dtype=np.float)

    def getUnixTime(self, units=False):
        """
            return the time of observation in seconds since epoch
        """
        if units: return np.array(self.hd1['UNIXTIME'], dtype=np.float) * u.s
        return np.array(self.hd1['UNIXTIME'], dtype=np.float)

    
    def getSpecUnit(self, apflag=True):
        """
            return unit of spectrum as string
            apflag: convert string to astropy units (default: True)            
        """
        spunit = self.hd1['TUNIT24'].replace('counts','count').replace('sec','s')
        print(spunit)
        if apflag: spunit = u.Unit(spunit) 
        return spunit
    
    def getVeloUnit(self, apflag=True):
        """
            return units of velocity axis as string
            apflag: convert string to astropy units (default: True)
        """
        vunit = self.hd1['TUNIT3']
        if apflag: vunit = u.Unit(vunit) 
        return vunit
    
    def getVlsr(self):
        """
            return the v_LSR (with units)
        """
        return np.array(self.hd1['VELOCITY'], dtype=np.float) * u.m/u.s
    
    def getDataVersion(self):
        """
            return the release version of the data
        """
        return np.array(self.hd1['LEVEL0'], dtype=np.float)
    
    def getLineSpecie(self):
        """
            return the observed species
        """
        return self.dd1['Line']
    
    def getLineID(self):
        """
            return the pixel identifier (for all 3 lines)
        """
        return self.dd1['TELESCOP']
    
    def getRestFrequencies(self):
        """
            return the rest frequency array (for all 3 lines)
        """
        return np.array(self.dd1['RESTFREQ'], dtype=np.float)
    
    def getBiasCurr(self):
        """
            return the bias current array (for all 3 lines)
        """
        return np.array(self.dd1['BiasCurr'], dtype=np.float)
    
    def getBiasVolt(self):
        """
            return the bias voltage array (for all 3 lines)
        """
        return np.array(self.dd1['BiasVolt'], dtype=np.float)
    
    def getTotalPower(self):
        """
            return the total power array (for all 3 lines)
        """
        return np.array(self.dd1['TotPower'], dtype=np.float)
    
    def getIFPower(self):
        """
            return the IF power array (for all 3 lines)
        """
        return np.array(self.dd1['IFPower'], dtype=np.float)
    
    def getBiasPot(self):
        """
            return the Bias Pot  (for all 3 lines)
        """
        return np.array(self.dd1['BiasPot'], dtype=np.float)
    
    def getPIDState(self):
        """
            return the PID state  (for all 3 lines)
        """
        return np.array(self.dd1['PIDState'], dtype=np.float)
    
    def getLNABias(self):
        """
            return the LNA bias (for all 3 lines)
        """
        return np.array(self.dd1['LNABias'], dtype=np.float)
    
    def getLNACurr(self):
        """
            return the LNA current (for all 3 lines)
        """
        return np.array(self.dd1['LNACurr'], dtype=np.float)
    
    def getBeamFac(self):
        """
            return the beam factor (for all 3 lines)
        """
        return np.array(self.dd1['BeamFac'], dtype=np.float)
    
    def getTsys(self):
        """
            return the system noise temperature
            (set to zero in Level <1)
        """
        return np.array(self.dd1['TSYS'], dtype=np.float)
    
    def getTrx(self):
        """
            return the receiver temperature  (for all 3 lines)
        """
        return np.array(self.dd1['Trx'], dtype=np.float)
    
    def getOutRange(self):
        """
            return the out-of-range status  (for all 3 lines)
        """
        return np.array(self.dd1['OutRange'], dtype=np.int)
    
    def getIntTime(self):
        """
            return the integration time  (for all 3 lines)
        """
        return np.array(self.dd1['IntTime'], dtype=np.float)
    
    def getNumOff(self):
        """
            return the number of OFF observations(?) (for all 3 lines)
        """
        return np.array(self.dd1['NumOff'], dtype=np.int)
    
    def getNPix(self):
        """
            return the number of OFF observations(?) (for all 3 lines)
        """
        return np.array(self.dd1['MAXIS1'], dtype=np.int)
    
    def getImageFrequencies(self):
        """
            return the image frequencies (for all 3 lines)
        """
        return np.array(self.dd1['IMAGFREQ'], dtype=np.float)
    
    def getDataLevel(self):
        """
            return the Data Level
        """
        return self.hd1['Level0']
    
    def getScanNumber(self):
        """
            return the Data Level
        """
        return np.int(self.hd1['SCAN'])
    
    def getObsID(self):
        """
            return the Data Level
        """
        return np.int(self.hd1['OBSID'])

    def getHistory(self):
        """
            get the History information in the Extension 1 header
        """
        hist = self.hd1['HISTORY']
        return hist

    def getComments(self):
        """
            get the History information in the Extension 1 header
        """
        cmnt = self.hd1['Comment']
        return cmnt
    
    
    def getL05Repair(self, line, chan=1, maskflag=False):
        """
        function to retrieve the information for the Level 0.5 despiked pixels
        line: 'C+', 'N+', or 'O I'
        chan: only used for 'N+' since we had two pixels: 1 and 2
        
        To do:  test for robustness since it evaluates the comment section of the header; 
                don't know what else is in this section.
        
        V. Tolls, SAO; 8/16/2018
        """
        if line=='C+':
            brd=3
            inp=1
        elif line=='N+':
            brd=1 
            inp=1
        elif line=='O I':
            brd=2
            inp=1
        
        cmnt = self.hd1['Comment']
        
        com = np.genfromtxt(cmnt, dtype=None, names=['brd', 'inp', 'chan', 'oldv', 'newv', 'line', 'pixel'],
                            delimiter=[3,3,6,12,13,5,5], skip_header=2)
        
        sel = np.where((com['brd']==brd)&(com['inp']==inp))
        
        if maskflag:
            mask = self.getLineMask()
            #for ii in com['chan'][sel]: mask[ii] = 1
            mask[com['chan'][sel]] = 2
            return np.array(com['chan'][sel]), mask
        else:
            return np.array(com['chan'][sel])


##################################################################################
# misc. functions to manipulate spectra .... some might need improvement
##################################################################################

def writeSTO2(ofile, dat, header):
    """
    Writing STO-2 style Level 1 FITS files that should be CLASS readable
    
    ofile:  name of output file; should end in .fits
    dat:    data structure for output data; updated version of the Level 0 data structure
    header: header for file; should be updated version of Level 0 header
    
    with respect to Lev ):
    updated header: OFFRA, OFFDEC, TCAL
    added header:   CALSCAN1, OFFSCAN1, CALSCAN2, OFFSCAN2, SEQRASTER, HOTSEQ1, HOTSEQ2
    updated bin table: CDELT2, CDELT3, TSYS,  INTTIME (->0 for OI), NUMOFF, DATA (now calibrated) 
    """
    
    
    # still needs development .....
    
    
    fits.writeto(ofile, dat, header)



def cleanSpec1D(dat, thresh = 500., verbose=False, getReport=False):
    """
    Remove spikes from a 1-D, single line spectrum
    Better use cleanSpecPeak() below!
    """
    
    ndat = dat.copy()
    fdat = butter_lowpass_filtfilt(ndat, 100, 10000)

    # try to clip the bad data    
    Ddat = ndat - fdat
    tag = mlab.find(np.abs(Ddat)<thresh)
    if verbose: print('median ndat/fdat/Ddat: ', np.median(ndat), np.median(fdat), np.median(Ddat))
    if verbose: print('tag: ', tag.tolist())
    
    report = []
    if tag.shape[0]>0:
        J = find(diff(tag)>1)
        nn = ndat.shape[0]
        for K in split(tag, J+1):
            # this is the array of data to be changed!
            # we are assuming isolated spikes and not spike city
            # the ends can be trimmed!
            if verbose: print('affected pixels: ', K)
            if ((K.min()>5)&(K.max()<nn-5)&(np.mean(Ddat[K])>thresh*0.7)):
                lw = np.arange(K.min()-4,K.min(),1)
                up = np.arange(K.max()+1,K.max()+5,1)
                updw = np.hstack([lw,up])
                # interpolate
                it = np.interp(K,updw,dat[updw])
                ndat[K] = it
                for kk in K:
                    rstr = 'cleaned pixel: %i %12.4f %14.2f'%(kk, ndat[kk], dat[kk])
                    report.append(rstr)
                if verbose: print('updw:  ', updw, dat[updw])
                if verbose: print('split: ', K, ndat[K], Ddat[K], it)

    if getReport: return ndat, report
    else: return ndat
    

def cleanSpecPeak1D(dat, thresh = 1E9, verbose=False, getReport=False, bflag=True, boff=2, vv=None, imethod='cubic'):
    """
    Remove spikes from a 1-D, single line spectrum
    
    getReport: return a short message about what was cleaned.
    bflag:     also clean the next pixel around the spike
    boff:      number of pixel to be taken off for wings (besides central pixel of spike)
    vv:        provide velocity information (velocity of affected pixels returned in report)
    method:    use a cubic spline interpolation (scipy.interpolation) or just linear interpolation (numpy.interp)
    """
    
    ndat = dat.copy()
    unit = ndat.unit
    ndat = ndat.value

    # try to clip the bad data    
    tag = np.argwhere(np.abs(ndat)>thresh)
    if verbose: print('tag: ', tag.tolist())
    
    # correcting neighboring pixels
    if bflag: 
        brange = np.arange(boff) + 1
    
    report = np.array([(0., 0., '', -1, 0.0, 0.0, 0.0)], dtype=[('scan', 'i4'),('obsid', 'i4'),('type', 'S10'),('pixel', 'i4'),('nval', 'f8'), ('val', 'f8'), ('velo', 'f4')])
    nrep = 0
    
    if tag.shape[0]>0:
        J = find(diff(tag)>1)
        nn = ndat.shape[0]
        for K in split(tag, J+1):
            # this is the array of data to be changed!
            # we are assuming isolated spikes and not spike city
            # the ends can be trimmed!
            if verbose: print('affected pixels: ', K)
            if ((K.min()>7)&(K.max()<nn-7)&(np.mean(ndat[K])>thresh*0.9)):
                if bflag:
                    K = np.hstack((K.min()-brange,K))
                    K = np.hstack((K,K.max()+brange))
                
                # interpolate
                if imethod=='cubic':
                    lw = np.arange(K.min()-6,K.min(),1)
                    up = np.arange(K.max()+1,K.max()+7,1)
                    updw = np.hstack([lw,up])
                    tck = interpolate.splrep(updw, dat[updw], s=0)
                    it = interpolate.splev(K, tck, der=0)
                else:
                    lw = np.arange(K.min()-4,K.min(),1)
                    up = np.arange(K.max()+1,K.max()+5,1)
                    updw = np.hstack([lw,up])
                    it = np.interp(K,updw,dat[updw])
                
                if verbose: print('updw:  ', updw, dat[updw])
                if verbose: print('split: ', K, ndat[K], it)
                ndat[K] = it
                for kk in K:
                    if vv==None:
                        #rstr = 'cleaned pixel: %i %12.4f %14.2f %7.2f'%(kk, ndat[kk], dat[kk], 9999.0)
                        rsa = np.array([(0., 0., '', kk, ndat[kk], dat[kk].value, 9999.0)], dtype=[('scan', 'i4'),('obsid', 'i4'),('type', 'S10'),('pixel', 'i4'),('nval', 'f8'), ('val', 'f8'), ('velo', 'f4')])
                        nrep += 1
                    else:
                        #rstr = 'cleaned pixel: %i %12.4f %14.2f %7.2f'%(kk, ndat[kk], dat[kk], vv[kk])
                        rsa = np.array([(0., 0., '', kk, ndat[kk], dat[kk].value, vv[kk].value)], dtype=[('scan', 'i4'),('obsid', 'i4'),('type', 'S10'),('pixel', 'i4'),('nval', 'f8'), ('val', 'f8'), ('velo', 'f4')])
                        nrep += 1
                    report = np.vstack((report,rsa))

    if nrep > 0: report = report[1:]
    else: report=None
    
    if getReport: return ndat*unit, report
    else: return ndat*unit



def removeSpike1D(idat, thresh):
    """
    simple spike removal - experimental!
    
    Input:
        thresh: threshold to determine pixels values to be replaced with median
       
    Ouput:
        modified data array

    """
    dat = idat.copy()
    med = np.median(dat)
    sel = np.where(dat>med+thresh)
    if len(sel[0])>0: dat[sel[0]] = med
    sel = np.where(dat<med-thresh)        
    if len(sel[0])>0: dat[sel[0]] = med
    
    return dat



def removeRawSpike1D(ixdat, iydat, thresh=1E6, nwpix=2, borderpix = 20, method='smooth', retbadpix=False, window_width=37, 
                     verbose=False, tbad=None, pbad=False, rmethod='replace'):
    """
    simple spike removal in raw spectrum - experimental!
    The spikes are very high!
    The bad pixels are determined using the scipy median filter "medfilt" with a window_width of 5 (changeable).
    The interpolation is done using function repPixInterp, which uses the scipy interpolation functionality
    
    Input:
        thresh: threshold to determine pixels values to be replaced (default: 1E6)
        borderpix:  number of pixels of either side of spike used for interpolation (not yet implemented)
        method:     method could be smoothing 'smooth' or median filtering 'medfilt'
        nwpix:      number of bad pixels in the wings to be repaired (minimum should be 1, default is 2)
        retbadpix:  if True, return the bad pixel array
        window_width: scalar size of median window width; should be odd number 
        rmethod:    'replace, 'nan'
       
    Ouput:
        modified data array

    """
    from scipy.signal import medfilt
    
    # remove the units (if present), which will be added back later
    try:
        xdat = ixdat.value.copy()
        ydat = iydat.value.copy()
    except:
        xdat = ixdat.copy()
        ydat = iydat.copy()
    
    rc = nwpix
    
    if method=='medfilt': out = medfilt(ydat, kernel_size=window_width)
    else: out = smoothData(ydat, cutoff=10, fs=100)
    
    badpix = np.squeeze(np.argwhere(np.abs(ydat-out)>thresh))
    if pbad:
        print(np.abs(ydat-out)[badpix])
        print(thresh[badpix])
    
    if np.isscalar(badpix): badpix = np.array(badpix)
    if verbose: print(method, badpix, badpix.size)
    if np.all(tbad)!=None:        
        if badpix.size>0:
            #print('shape: ', tbad.shape, badpix.shape)
            #print('arrays: ', tbad, badpix)
            badpix = np.hstack((badpix,tbad))
            badpix = np.sort(badpix)
            badpix = np.unique(badpix)
        else:
            badpix = tbad
        
    #if verbose: print('rep badpix: ', np.squeeze(badpix))
    
    if badpix.size>0:
        if badpix.size==1:
            bp = np.arange(badpix-rc, badpix+rc+1, 1)
        else:
            bp = np.arange(badpix[0]-rc, badpix[0]+rc+1, 1)
        for i in range(1,badpix.size):
            bp = np.vstack([bp, np.arange(badpix[i]-rc, badpix[i]+rc+1, 1)])
        bp = np.unique(bp)
        bp = bp[np.where((bp>=0) & (bp<xdat.size))]
    
        # rather than reversing the order of the array, we multiply the (unused) x-array with -1., which essentially reverses the 
        # order as well, but with less impact.
        if np.all(np.diff(xdat)<0): fac = -1.
        else: fac = 1.
        
        #if verbose: print('rep: ', np.squeeze(bp))
        if rmethod=='nan':
            if bp.size>0: ydat[bp] = np.nan
        elif rmethod=='replace':
            if bp.size>0: ydat = repPixInterp(fac*xdat, ydat, bp)
        else:
            if bp.size>0: ydat = repPixInterp(fac*xdat, ydat, bp)
    else:
        bp=np.empty(shape=(0))
    
    try:
        xdat = xdat*ixdat.unit
        ydat = ydat*iydat.unit
    except:
        pass
    
    if retbadpix: return xdat, ydat, bp
    else: return xdat, ydat


from itertools import groupby, cycle 
  
def groupSequence(l): 
    temp_list = cycle(l) 
    
    next(temp_list) 
    groups = groupby(l, key = lambda j: j + 1 == next(temp_list)) 
    for k, v in groups: 
        if k: 
            yield tuple(v) + (next((next(groups)[1])), ) 



def addHeaderBadPixels(icom, imask):
    """
    Add the bad pixels listed in the header in the comment section as determined by scooper, the Level 0 to Level 0.5 program.
    The flag value for bad pixels here is 2.
    
    However, the header information should have been added when creating the mask! If it is added again, it should not change 
    the already existing information in case the functions has been invoked more than once.
    """
    nmask = imask.copy()
    try:
        #icom = hd1['COMMENT']
        com = np.genfromtxt(icom, dtype=None, names=['brd', 'inp', 'chan', 'oldv', 'newv', 'line', 'pixel'],
                            delimiter=[3,3,6,12,13,5,5], skip_header=2)

        # bit 1 => add 2 to mask: originally despiked in scooper
        # first NII line, second NII line, and [OI] line
        nmask[0,com['chan'][np.where((com['brd']==1)&(com['inp']==1))]-1] = np.bitwise_or(nmask[0,com['chan'][np.where((com['brd']==1)&(com['inp']==1))]-1], 2)
        nmask[1,com['chan'][np.where((com['brd']==1)&(com['inp']==2))]-1] = np.bitwise_or(nmask[1,com['chan'][np.where((com['brd']==1)&(com['inp']==2))]-1], 2)
        nmask[2,com['chan'][np.where((com['brd']==3)&(com['inp']==1))]-1] = np.bitwise_or(nmask[2,com['chan'][np.where((com['brd']==3)&(com['inp']==1))]-1], 2)
        if verbose: print('2: ', com['chan'][np.where((com['brd']==3)&(com['inp']==1))]-1)
    except:
        pass
    return nmask


def repSpike1D(ispec1, imask1, sclip=0, eclip=0, thresh = 10000, near=1, bit2flag=False, 
               verbose=False, badpix=None, method='cubicspline'):
    """
    This function is still experimental, hence the many print statements for debugging.
    
    This is the latest cleaning function, based on a several step cleaning procedure.
    The first step removes and replaces the large spikes and tries to interpolate the missing
    values. There is an option for replacing the first sclip pixels with the value of pixel[sclip] 
    and the last eclip pixels with the value of pixel[eclip]. These ranges were almost impossible 
    to clean and repair for [NII], and this methods seems to be the best option since the data are 
    useless in any case. 
    The second step removes smaller spikes that have an amplitude larger than thresh compared 
    to the adjacent pixels. Here are 2 option, using the median filtered difference or the ALS difference
    of old minus filtered spectrum? Only positive spikes are removed. Only median filer used right now.
    
    Compare to subroutine setMask() below for changes in flagging:
    Meaning of mask bits (might change slightly in future by adding new values, 16 bits available):    
        bit 0 => add  1 to mask:  all spectral data values are NaNs or zeros
        bit 1 => add  2 to mask:  spikes listed in comment area of raw (level 0.5+) files and badpix pixels
        bit 2 => add  4 to mask:  new despiking, e.g., residuals of transmission LO interference or so
        bit 3 => add  8 to mask:  end of spectrum outliers clipped spectral elements
        bit 4 => add 16 to mask:  forced bad pixels provided with badpixel
    
    Inputs:
        ispec1:      "dirty" spectrum
        imask1:      initial mask, most likely empty
        sclip:       number of pixels to be replaced at beginning of spectrum
        eclip:       number of pixels to be replaced at end of spectrum
        thresh:      threshold for identifying smaller spikes 
                     (value independent of absolute pixel value; 10000 too aggressive?) 
        near:        number of adjacent pixels also to be masked (default: 1)
        bit2flag:    True: interpolate the value of pixels masked in bit 2 (a.k.a., the pixels identified in the file header comment section)
                     False: ignore...
        badpix:      user selected additional pixels to be mask (default is None)
                     example is: Eta Car 2, all pixels between 479 and 491 should be masked by default since bad
        method:      method for interpolation: splrep or cubicspline
    Function created after test notebook: notebooks/STO2/EtaCar/EtaCar2_spike_cleaning_single_spectrum_v4_testing.ipynb
        
    initially created: 8/21/2019, VTO
    """
     
    ### Step 1
    
    # identify the large spikes and set mask
    # set bit 3 for "end of spectrum outliers"
    dtt = type(ispec1)
    if type(ispec1)==type(np.zeros(2)):
        spec1 = ispec1.copy()
    else:
        spec1 = ispec1.value
    mask1 = imask1.copy()
    if verbose: print(spec1.shape)
    
    # Step 1
    # clip ends of scan first in case there are major spikes.
    # set bit 3
    if sclip>0:
        #sclip = 70
        # replace the beginning sclip pixels with the value in the edge pixel
        if spec1[sclip] < 0.5E9:
            spec1[:sclip] = np.ones(sclip) * spec1[sclip]
            # set the mask
            mask1[:sclip] = np.bitwise_or(mask1[:sclip], 8)   
        else:
            print('Error. repSpike1D: spectrum in pixel sclip is spike.')
        
    if np.abs(eclip)>0:
        #eclip = -100
        
        # replace the eclip pixels at end of scan with the value at its edge
        if spec1[eclip] < 0.5E9:
            spec1[eclip:] = np.ones(np.abs(eclip)) * spec1[eclip]
            # set the mask
            mask1[eclip:] = np.bitwise_or(mask1[eclip:], 8)   
        else:
            print('Error. repSpike1D: spectrum in pixel eclip is spike.')
            
    # Step 2
    # Identify and mask major spikes with counts over 5E8, set to nan.
    # set bit 2 for "new de-spiking"
    args = np.argwhere(spec1 > 0.5e9)
    for arg in args:
        spec1[arg] = np.nan
        mask1[arg] = np.bitwise_or(mask1[arg], 4)
        if arg<spec1.size-1:    
            spec1[arg+1] = np.nan
            mask1[arg+1] = np.bitwise_or(mask1[arg+1], 4)
            if arg<spec1.size-2:    
                spec1[arg+2] = np.nan
                mask1[arg+2] = np.bitwise_or(mask1[arg+2], 4)
        if arg>1:
            spec1[arg-1] = np.nan
            mask1[arg-1] = np.bitwise_or(mask1[arg-1], 4)
            if arg>2:
                spec1[arg-2] = np.nan
                mask1[arg-2] = np.bitwise_or(mask1[arg-2], 4)
    idx = np.arange(spec1.size)
    
    # add the badpix if it is not None:
    if badpix is not None:
        obadpix  = np.array(badpix, dtype=np.int)
        mask1[obadpix] = np.bitwise_or(mask1[obadpix], 16)
        
    
    try:
        # get the nan groups:
        midx = idx[np.where(mask1==4)]
        gmidx = list(groupSequence(midx))

        seps = [gmidx[0][0]]
        for i in range(len(gmidx)-1): seps.append(gmidx[i+1][0] - gmidx[i][-1])
        seps.append(spec1.size - gmidx[-1][-1])
    except:
        pass
    
    #if verbose: print('Separations of groups: ', seps)
    
    # create an index array for the spectrum (e.g., replacement for velocity array)
    pidx  = np.arange(spec1.shape[0])
    pidx1 = np.arange(spec1.shape[0])
    
    # group all pixels with mask flag 4 (or flag2 and flag 4 but not flag 8)
    mpidx1 = pidx[np.where(np.bitwise_and(mask1, 4))]
    if bit2flag:
        mpidx1 = pidx[(np.bitwise_and(mask1, 2)|np.bitwise_and(mask1, 4))&(np.bitwise_and(mask1, 8))]
    if mpidx1.size>0:
        gmidx1 = list(groupSequence(mpidx1))
        n_gmidx1 = len(gmidx1)
        
        seps1 = [gmidx1[0][0]]
        for i in range(len(gmidx1)-1): seps1.append(gmidx1[i+1][0] - gmidx1[i][-1])
        seps1.append(spec1.size - gmidx1[-1][-1])
    else:
        n_gmidx1 = 0
        
    # interpolation parameters
    s=0.0   # smoothing condition, if 0.0, do interpolation if no weights provided
    k=3     # order of spline fit -> use cubic splines
    
    for i in range(n_gmidx1):
        g0 = gmidx1[i]   # get i-th group
        ps = g0[0]       # first element in group
        pe = g0[-1]      # last element in group
        ds = 30          # repair interval on start side
        if ds > seps1[i]: ds = seps1[i]
        de = 30          # repair interval on end side
        if de > seps1[i+1]: de = seps1[i+1]
        ds = int(ds)
        de = int(de)
        
        # the arrays below exclude ps to pe, the range we want to interpolate
        isp = np.hstack((spec1[ps-ds:ps], spec1[pe+1:pe+de]))   # spectrum
        ipi = np.hstack((pidx1[ps-ds:ps], pidx1[pe+1:pe+de]))   # pixel index array
        
        # interpolate missing/masked values
        if isp.size>3:
            if method=='splrep':
                it = interpolate.splrep(ipi, isp, k=k)
                spec1arep = interpolate.splev(pidx[ps:pe+1], it, der=0)
                spec1[ps:pe+1] = spec1arep
            if method=='cubicspline':
                cs = interpolate.CubicSpline(ipi, isp)
                spec1[ps:pe+1] = cs(pidx[ps:pe+1])
            if method=='polynomial':
                z = np.polyfit(ipi, isp, 5)
                pf = np.poly1d(z)
                spec1[ps:pe+1] = pf(pidx[ps:pe+1])
        else:
            print('ERROR: no interpolation of major spikes....')
            print(ipi.shape, isp.shape, g0)

    ##### Step 3
    # Remove additional smaller spikes
    # set bit 2 for "new de-spiking"
    
    # median filter option
    ks = 7         # median filter kernel
    out2 = medfilt(spec1, kernel_size=ks)
    
    # ALS option
    # out3 = als(spec1, lam=0.1, p=0.1)

    #thresh = 60000    # spike threshold, spikes below threshold can be removed later.
    tsel = np.argwhere((spec1 - out2)>thresh)
    # plt.plot(idx[tsel],spec1[tsel],'*')
    # mask the small spikes with 4
    mask1[tsel] = np.bitwise_or(mask1[tsel], 4)
    # remove pixel before and after since we do not know if they are affected.
    if (near > 0) & (tsel.size > 0):
        for ne in range(1,near+1,1):
            tsel1 = tsel[np.where((tsel>ne)&(tsel<1024-ne))]
            mask1[tsel1-ne] = np.bitwise_or(mask1[tsel1-ne], 4)
            mask1[tsel1+ne] = np.bitwise_or(mask1[tsel1+ne], 4)
    if verbose: print('small spikes (bit 2, flag 4) at (+/-1): ', np.squeeze(tsel))
    
    ##################
    # now, try to interpolate the identified pixels
    
    mpidx1 = pidx[np.argwhere(np.bitwise_and(mask1, 4))]
    n_mpidx1 = len(mpidx1)
    
    s=0.0
    k=3
    
    for i in range(n_mpidx1):
        e0 = int(mpidx1[i])   # get i-th element
        #print(i, e0)
        ds = int(20)
        if ds > e0: ds = e0
        de = int(20)
        if de > spec1.size: de = int(spec1.size - e0 - 1)
        
        # the arrays below exclude ps to pe, the range we want to interpolate
        e0 = int(e0)
        isp = np.squeeze(spec1[e0-ds:e0+de])
        ipi =  np.squeeze(pidx1[e0-ds:e0+de])
        ims = np.squeeze(mask1[e0-ds:e0+de])
        esel = np.argwhere(np.bitwise_and(ims, 4)==0)
        esel2 = np.squeeze(np.argwhere(np.bitwise_and(ims, 4)))
        ispr = np.squeeze(isp[esel])
        ipir = np.squeeze(ipi[esel])
        imsr = np.squeeze(ims[esel])
        #print(e0, ds, esel2, e0-ds+esel2)
    
        try:
            if method=='splrep':
                it = interpolate.splrep(ipir, ispr, k=k)
                sprep = np.squeeze(interpolate.splev(ipi[esel2], it, der=0))
            if method=='cubicspline':
                cs = interpolate.CubicSpline(ipir, ispr)
                sprep = cs(ipi[esel2])
            if method=='polynomial':
                z = np.polyfit(ipir, ispr, 5)
                pf = np.poly1d(z)
                sprep = pf(ipi[esel2])
                

            #print('   org: ', isp[esel2])
            #print(isp)
            isp[esel2] = sprep
            #print(isp)
            #print('   rep: ', spec1a[e0-ds+esel2])
            spec1[e0-ds+esel2] = sprep
        except:
            print('error in interpolate function: ', ispr.size, ispr)
            print(ipir)

    return spec1, mask1



def repPix1D(dat, rpix):
    
    return repPix(dat, rpix)



def repPix(dat, rpix):
    """
    replacing pixel values in spectra with interpolated values from neighboring pixels.
    This function uses numpy interp which is only a linear interpolation!
    
    (Note: the pixels can also be repaired when reading the data from file whereas this
    function creates a deep copy of the data array, which is then repaired.)
    
    Input:
        dat: array of spectra (1-D to 3-D) with last index for pixels with unit
        rpix: array of pixels to be "repaired", 
             referring to the pixels in the last index of the array!
             E.g.: dat.shape=[2,4,1024] is an [2,4] array of 1024-pixel spectra
       
    Ouput:
        modified deepcopy of input data array
    """
    ndim = dat.ndim
    ndat = dat.copy()
    try:
        unit = ndat.unit
        ndat = ndat.value
    except:
        # assume we have no unit
        unit = None
        #ndat = ndat
        

    if ndim==1:
        nn = dat.shape[0]
        mask = np.ones(nn, dtype=bool)
        mask[rpix] = False
    
        # interpolate
        it = np.interp(rpix,np.arange(nn)[mask],dat[mask])
        ndat[rpix] = it
        
        return ndat*unit
    elif ndim==2:
        nn = dat.shape
        mask = np.ones(nn[1], dtype=bool)
        mask[rpix] = False
        
        for i in range(nn[0]):
            # interpolate
            sspec = ndat[i,:]
            it = np.interp(rpix,np.arange(nn[1])[mask],sspec[mask])
            ndat[i,rpix] = it
        
        return ndat*unit 
    elif ndim==3:       
        nn = dat.shape
        mask = np.ones(nn[2], dtype=bool)
        mask[rpix] = False
        
        for i in range(nn[0]):
            for j in range(nn[1]):
                # interpolate
                sspec = ndat[i,j,:]
                it = np.interp(rpix,np.arange(nn[1])[mask],sspec[mask])
                ndat[i,j,rpix] = it
        if unit!=None:
            return ndat*unit 
        else:
            return ndat
    else:
        print('Error: repPix cannot handle arrays with dimension>3.')
        print('Error: returning unchanged array.')
        return dat



def repPixInterp(xdat, ydat, rpix, s=0.0, k=3):
    """
    replacing pixel values in spectra with interpolated values from neighboring pixels.
    This function uses scipy interpolate with cubic spline.
    (Note: the pixels can also be repaired when reading the data from file whereas this
    function creates a deep copy of the data array, which is then repaired.)
    
    Input:
        (xy)dat: array of spectra (1-D to 3-D) with last index for pixels with unit
        rpix: array of pixels to be "repaired", 
             refering to the pixels in the last index of the array!
             E.g.: dat.shape=[2,4,1024] is an [2,4] array of 1024-pixel spectra
       
    Ouput:
        modified deepcopy of input data array
    """
    
    #strip units
    rpix = rpix.copy()
    nxdim = xdat.ndim
    nxdat = xdat.copy()
    nydim = ydat.ndim
    nydat = ydat.copy()
    try:
        yunit = nydat.unit
        nydat = nydat.value
    except:
        # assume we have no unit
        yunit = None
        #ndat = ndat
    try:
        xunit = nxdat.unit
        nxdat = nxdat.value
    except:
        xunit = None
    
    

    if nydim==1:
        # we have to make sure the xdata are in ascending order for interpolation
        #asc = 1 if np.mean(nxdat[1:]-nxdat[:-1])>0 else -1
        asc = 1 if np.all(np.diff(nxdat))>0 else -1
        nxdat = nxdat[::asc]
        nydat = nydat[::asc] 
        if asc==-1: rpix = 1023 - rpix
        
        nn = ydat.shape[0]
        # create mask for good data points (True)
        mask = np.ones(nn, dtype=bool)
        mask[rpix] = False
    
        # interpolate
        # Note: if interpolate throws an error on input data, check if nxdat is in strictly ascending order.
        it = interpolate.splrep(nxdat[mask], nydat[mask], s=s, k=k)
        nydat[rpix] = interpolate.splev(nxdat[rpix], it, der=0)
        
        if yunit!=None: return nydat[::asc]*yunit
        else: return nydat[::asc]
        
    elif nydim==2:
        # we have to make sure the xdata are in ascending order for interpolation
        #asc = 1 if np.mean(nxdat[0,1:]-nxdat[0,:-1])>0 else -1
        asc = 1 if np.all(np.diff(nxdat))>0 else -1
        nxdat = nxdat[:,:asc]
        nydat = nydat[:,:asc] 
        if asc==-1: rpix = 1023 - rpix
        
        nn = ydat.shape
        mask = np.ones(nn[1], dtype=bool)
        mask[rpix] = False
        
        for i in range(nn[0]):
            # interpolate
            if nxdat.ndim==2: sxx = nxdat[i,:]
            else: sxx = nxdat
            sspec = nydat[i,:]
            #it = np.interp(rpix,np.arange(nn[1])[mask],sspec[mask])
            #ndat[i,rpix] = it
            it = interpolate.splrep(sxx[mask], sspec[mask], s=0, k=3)
            nydat[i,rpix] = interpolate.splev(sxx[rpix], it, der=0)
        
        if yunit!=None: return nydat[:,:asc]*yunit 
        else: return nydat[:,:asc]
    elif nydim==3:       
        # we have to make sure the xdata are in ascending order for interpolation
        asc = 1 if np.mean(nxdat[0,0,1:]-nxdat[0,0,:-1])>0 else -1
        nxdat = nxdat[:,:,asc]
        nydat = nydat[:,:,asc] 
        if asc==-1: rpix = 1023 - rpix
        
        nn = ydat.shape
        mask = np.ones(nn[2], dtype=bool)
        mask[rpix] = False
        
        for i in range(nn[0]):
            for j in range(nn[1]):
                # interpolate
                if nxdat.ndim==3: sxx = nxdat[i,j,:]
                else: sxx = nxdat
                sspec = nydat[i,j,:]
                #it = np.interp(rpix,np.arange(nn[1])[mask],sspec[mask])
                #nydat[i,j,rpix] = it
                it = interpolate.splrep(sxx[mask], sspec[mask], s=0, k=3)
                nydat[i,rpix] = interpolate.splev(sxx[rpix], it, der=0)
        if yunit!=None: return nydat[:,:,asc]*yunit 
        else: return nydat[:,:,asc]
    else:
        print('Error: repPix cannot handle arrays with dimension>3.')
        print('Error: returning unchanged array.')
        return ydat
        

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    from: http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray



def getFiles(stobsid, enobsid, cdirnum, path):
    afiles = []
    for i in range(stobsid, enobsid, 1):
        aa = os.path.join(path,cdirnum,'OTF%s_%05i.fits'%(cdirnum,i))
        # check if exists and append
        if os.path.isfile(aa): afiles.append(aa)
    return afiles



def resampleSpectrumWMask(ivv, isp, mask, step, verbose=False):
    """
    resampling the velocity axis (vv) and the intensity (spec) in increments of channels (step) by 
    averaging the velocity channels (average is center of new channel) and
    averaging the intensity channels (average to preserve the integrated intensity
    
    Input:
        ivv: array of velocities v (with constant delta v), can be 1-D or same dimension as spectrum (with last dimension as spectral axis)
        isp: array of intensities spec (unit does not matter), can be multidimensional, but last dimension should be spectral axis
             e.g. if spectrum has 1024 channels, allowed dimensions include (isp.shape): (1024,), (m, 1024), (n, m, 1024)
        mask: the mask array associated with the data above of the same dimensions
        step: number of channels (integer!) summed for resampled spectrum
      
    Output: nvv, nspec
        nvv: new velocity array
        nspec: new spectrum
    """
    # if step=1, no resampling required, return input array
    if step==1:
        return ivv, isp, mask
    vvsh = list(ivv.shape)
    # if required, adjust the dimension of the spectral axis
    if vvsh[-1]%step > 0:
        ivv = ivv[...,0:vvsh[-1]-vvsh[-1]%step]
        isp = isp[...,0:vvsh[-1]-vvsh[-1]%step]
        vvsh = list(ivv.shape)
    vvsh[-1] = vvsh[-1]//step
    nvv = bin_ndarray(ivv, new_shape=vvsh, operation='mean')
    spsh = list(isp.shape)
    spsh[-1] = spsh[-1]//step
    nsp = bin_ndarray(isp, new_shape=spsh, operation='mean')
    nmask = resampleMask(mask, step)
    
    return nvv, nsp, nmask



def resampleSpectrum(ivv, isp, step, verbose=False):
    """
    resampling the velocity axis (vv) and the intensity (spec) in increments of channels (step) by 
    averaging the velocity channels (average is center of new channel) and
    averaging the intensity channels (average to preserve the integrated intensity
    
    Input:
        ivv: array of velocities v (with constant delta v), can be 1-D or same dimension as spectrum (with last dimension as spectral axis)
        isp: array of intensities spec (unit does not matter), can be multidimensional, but last dimension should be spectral axis
             e.g. if spectrum has 1024 channels, allowed dimensions include (isp.shape): (1024,), (m, 1024), (n, m, 1024)
        step: number of channels (integer!) summed for resampled spectrum
      
    Output: nvv, nspec
        nvv: new velocity array
        nspec: new spectrum
    """
    # if step=1, no resampling required, return input array
    if step==1:
        return ivv, isp
    vvsh = list(ivv.shape)
    # if required, adjust the dimension of the spectral axis
    if vvsh[-1]%step > 0:
        ivv = ivv[...,0:vvsh[-1]-vvsh[-1]%step]
        isp = isp[...,0:vvsh[-1]-vvsh[-1]%step]
        vvsh = list(ivv.shape)
    vvsh[-1] = vvsh[-1]//step
    nvv = bin_ndarray(ivv, new_shape=vvsh, operation='mean')
    spsh = list(isp.shape)
    spsh[-1] = spsh[-1]//step
    nsp = bin_ndarray(isp, new_shape=spsh, operation='mean')
    
    return nvv, nsp


def createMask(nn):
    """
    Create a mask of size nn
    """
    return np.zeros((nn), dtype=np.int32)


def setMask(imask, idx, val):
    """
    Set the pixels of a mask using OR: newvalue = oldvalue or val.
    The value should be according to the bit set: bit0: 1, bit2: 2, bit3: 4, ...
    
    Inputs:
        imask: mask to be updated
        idx:   pixel indices to be changed
        val:   value (bit value(s)!) for update
    
    Meaning of mask bits (might change slightly in future by adding new values, 16 bits available):    
        bit 0 => add 1 to mask:  all spectral data values are NaNs or zeros
        bit 1 => add 2 to mask:  spikes listed in comment area of raw (level 0.5+) files.
        bit 2 => add 4 to mask:  new despiking
        bit 3 => add 8 to mask:  end of spectrum outliers clipped spectral elements

    """
    if val is np.nan:
        print('Error: no mask value provided.  Nothing changed!!')
        return imask
    nm = np.zeros((imask.size), dtype=np.int)
    idx = np.array(idx, dtype=np.int)
    nm[idx] = val
    return np.bitwise_or(imask, nm)


def issetMask(mask, flag):
    """
    test if mask is set.
    Input:
       mask:    mask (integer array)
       flag:    flag bit (1,2,4,8,...) to check if set
       
    Output:
        list of arguments that are set
    """
    return np.squeeze(np.argwhere(np.bitwise_and(mask, flag)))
    


def resampleMask(mask, resp):
    """
    resampling the mask by combining resp number of pixels values bitwise
    input:
        mask: the full mask, 1d or 2d
        resp: number of pixels combined to new pixel
    
    WARNING: not yet fully tested (or used)
    """
    if mask.ndim==1:
        imask = np.squeeze(mask[:resp*(mask.size//resp)])
        nmask = imask[::resp]
        for i in range(1,resp,1):
            nmask = np.bitwise_or(nmask, imask[i::resp])
        
        return nmask
    elif mask.ndim==2:
        imask = np.squeeze(mask[:resp*(mask.shape[0]//resp),:])
        nmask = imask[::resp]
        for i in range(1,resp,1):
            nmask = np.bitwise_or(nmask, imask[i::resp,:])
        
        return nmask
    else:
        return None



def plotMask(mm1, yl=None, alp = 0.2, xx=None, singlecolor=False):
    """
    plot the masks, color coded according to which bit is set
    
    mm1:        mask data array
    yl:         y-axis min and max for plotting shaded region
    xx:         can be velocity array or such
    alp:        alpha for shaded region of mask
    """
    if singlecolor:
        cl = ['darkorange','darkorange','darkorange','darkorange','darkorange','darkorange']
    else:
        cl = ['yellow','darkorange','purple','blue','yellowgreen','darkolivegreen']
    if np.all(xx == None):
        xx = np.arange(mm1.size)
    if yl is None:
        try:
            yl = plt.ylim()
        except:
            yl = plt.gca().ylim()
    md = np.mean(np.diff(xx))
    # first bit  with value  1
    for px in np.argwhere(mm1&0b0001>0).flatten(): plx = plt.fill_between([xx[px],xx[px]+md],yl[1],yl[0], color=cl[0], alpha=alp, zorder=1)
    # second bit with value  2
    for px in np.argwhere(mm1&0b0010>0).flatten(): plx = plt.fill_between([xx[px],xx[px]+md],yl[1],yl[0], color=cl[1], alpha=alp, zorder=1)
    # third bit  with value  4
    for px in np.argwhere(mm1&0b0100>0).flatten(): plx = plt.fill_between([xx[px],xx[px]+md],yl[1],yl[0], color=cl[2], alpha=alp, zorder=1)
    # fourth bit with value  8
    for px in np.argwhere(mm1&0b1000>0).flatten(): plx = plt.fill_between([xx[px],xx[px]+md],yl[1],yl[0], color=cl[3], alpha=alp, zorder=1, step='pre')
    # fifth bit  with value 16
    for px in np.argwhere(mm1&0b10000>0).flatten(): plx = plt.fill_between([xx[px],xx[px]+md],yl[1],yl[0], color=cl[4], alpha=alp, zorder=1, step='pre')
    # sixth bit  with value 32
    for px in np.argwhere(mm1&0b100000>0).flatten(): plx = plt.fill_between([xx[px],xx[px]+md],yl[1],yl[0], color=cl[5], alpha=alp, zorder=1, step='pre')




def decontamSpec(repspec, cleanspec, xx, reprange, repborderwidth):
    """
       Function to remove emission from spectrum by replacing from rescaled, clean spectrum
       input:
            repspec: 1-d spectrum to be repaired
            cleanspec: 1-d clean spectrum
            xx: x-axis valid for both spectra (can be anything, velocity, frequency, or wavelength
            reprange: xx-range to be repaired
            repborderwidth: xx-range outside of reprange used for scaling
        output:
            1-d emission-removed spectrum
    """
    ruef_sm = smoothData(repspec)
    huf_sm = smoothData(cleanspec)
    # range for replacement is between -40 and 5 km/s
    # ruef needs to be repaired
    # use huf for repair
    rrange = [-40.,0.]
    sidewidth = 30.
    vsel = np.where(((xx>rrange[0]-sidewidth)&(xx<rrange[0]))|((xx>rrange[1])&(xx<rrange[1]+sidewidth)))[0]
    druef = ruef_sm[vsel]
    dhuf = huf_sm[vsel]
    fxx = xx[vsel]
    isel = np.where((xx>rrange[0])&(xx<rrange[1]))[0]
    nxx = xx[isel]
    rat = druef/dhuf
    # linear interpolation
    #f = np.interp(nxx,fxx,rat)
    # non-linear interpolation
    from scipy.interpolate import interp1d
    f2 = interp1d(fxx, rat, kind='slinear')
    f = f2(nxx)

    newspec = repspec.copy()
    newspec[isel] = cleanspec[isel] * f
    #asel = np.where((xx>rrange[0]-sidewidth)&(xx<rrange[1]+sidewidth))[0]
    
    return newspec


def smrepMask(xxi, yyi, mmi):
    # apply the mask
    yyr = yyi.copy()
    yyim = yyi.copy()
    yyim[np.where(mmi>0)] = np.nan

    if xxi[0]<xxi[-1]:
        fc = 1
        xx = xxi[np.isfinite(yyim)]
        yy = yyim[np.isfinite(yyim)]
    else:
        fc = -1
        xx = (xxi[np.isfinite(yyim)])[::-1]
        yy = (yyim[np.isfinite(yyim)])[::-1]
        
    try:
        spl = UnivariateSpline(xx, yy)
        spl.set_smoothing_factor(20000.0)
    
        rsel = np.where(np.isnan(yyim))
        yyr[rsel] = spl(xxi[rsel])  
    except:
        pass
    
    return yyr



def smoothData(dat, cutoff=100, fs=10000):
    """
        Butterworth filter data smoothing
        
        for more info see notebook: STO2_cleaning_data
        or scipy documentation
        
        cutoff: cutoff frequency
        fs:     sampling rate
    """
    return butter_lowpass_filtfilt(dat, cutoff, fs)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y



def getScanList():
    converters = {6: lambda s: float(s or 0)}
    ifile = '/rdat/Projects/STO-2/Data/data_new_all_v3.txt'
    dat = np.genfromtxt(ifile, dtype=None, names=['scan','obsid','target','type','ot_row','obstime','l','b',
                                                  'l_off','b_off','date','synCII','biasv','biasc','tpower'],
        delimiter=[6,7,16,8,7,12,12,11,9,10,25,11,9,11,9], autostrip=True, skip_header=2)

    return dat



def mkRep(bp, vel, spe, hd, typ, verbose=False):
    """
    bp: bad pixels for report
    vel: velocity array
    spe: spectrum
    hd:  header
    typ: spectrum type
    """
    if verbose: print(typ, bp[0], spe[bp[0]].value, 0.0, vel[bp[0]].value)
    
    rep0 = np.array([(0., 0., '', -1, 0.0, 0.0, 0.0)], dtype=[('scan', 'i4'),('obsid', 'i4'),('type', 'S10'),('pixel', 'i4'),('nval', 'f8'), ('val', 'f8'), ('velo', 'f4')])

    # create first report row
    if bp!=None:
        if len(bp)>0:
            rep = [np.array([(hd['scan'], hd['obsid'], typ, bp[0], spe[bp[0]].value, 0.0, vel[bp[0]].value)], 
                  dtype=[('scan', 'i4'),('obsid', 'i4'),('type', 'S10'),('pixel', 'i4'),('nval', 'f8'), ('val', 'f8'), ('velo', 'f4')])]
    
        # if there are more report rows, append them to the first row
        if len(bp)>1:
            for i in range(1,len(bp),1):
                rep.append(np.array([(hd['scan'], hd['obsid'], typ, bp[i], spe[bp[i]].value, 0.0, vel[bp[i]].value)], 
                  dtype=[('scan', 'i4'),('obsid', 'i4'),('type', 'S10'),('pixel', 'i4'),('nval', 'f8'), ('val', 'f8'), ('velo', 'f4')]))
            
        return rep
    else:
        return None



def top3perc(dat, lim=0.09, verbose=False):
    """
    sorting and returning the args for the highest 3 percent (or what set with lim)
    of values in dat. Does not work on 2-d arrays!
    dat: should be a 1d-array, e.g. a single spectrum
    """
    fdat = dat.flatten()
    nn = fdat.shape[0]
    
    # sorting in increasing order, highest values at end
    ss = np.argsort(fdat, axis=None)
    en = int(-lim*nn)
    if verbose: print('higest entries in data array (indices/values):       ', en, dat[ss[en:]])
    
    # return indices of highest values
    return ss[int(-lim*nn):]



# def getTsysOld(chdirnum, lin, verbose=False, ipath=None):
#     """
#         derive the system noise temperature from the 
#         first hot-load and first reference (sky) observation
#         input: 
#             chdirnum:  scan number of STO2 observations
#             lin:       desired line
#     """
#     
#     badpix = [480,481,482,483,484,485,486,487,488,489,490]
#     # determine Tsys
#     # otf cal hot load measurement; take the first HOT... file, although there should only be one
#     if ipath!=None: hname = os.path.join(ipath,chdirnum,'*HOT*.fits')
#     else: hname = os.path.join('./',chdirnum,'*HOT*.fits')
#     hfiles = glob.glob(hname)
#     if verbose: print('Cal Hot load: ', hfiles[0])
#     lvv, hot, lpos, lh1 = readSTO2(hfiles[0], lin, verbose=verbose, rclip=2)
#     
#     hot = repPix(hot, badpix)
#     if cleanflag: 
#         hot, hrep = cleanSpecPeak(hot, thresh=thresh, getReport=True, vv=lvv, boff=2)
#         rep = repUpdate(rep, hrep, np.int(lh1['SCAN']), np.float(lh1['OBSID']), 'hot')
# 
#     lTint = np.float(lh1['OBSTIME'])
#     Thot = np.float(lh1['CONELOAD'])
#     Tsky = 45.
# 
#     # otf cal ref measurement; take the first HOT... file, although there should only be one
#     hrname = os.path.join(ipath,chdirnum,'*REF*.fits')
#     hrfiles = glob.glob(hrname)
#     if verbose: print('Cal Ref load: ', hrfiles[0])
#     hrv, hrf, hrpos, hrh1 = readSTO2(hrfiles[0], lin, verbose=verbose, rclip=2)
#     hrf = repPix(hrf, badpix)
#     rprrep = mkRep(badpix, hrv, hrf, hrh1, 'load')
#     hrTint = np.float(hrh1['OBSTIME'])
#     if cleanflag: 
#         hrf, rrep = cleanSpecPeak(hrf, thresh=thresh, getReport=True, vv=hrv, boff=2)
# 
# 
#     #Tsys2 = 1. * xThot * ref / (xhot - ref)
#     y = hot / hrf
#     Tsys = (Thot - y * Tsky) / (y - 1.) 
#     
#     return Tsys, lvv

def getTsysLine(chdirnum, lin, rep=None, verbose=False, Tsky=45.*u.K, exTsys=1600.*u.K, cleanflag=False, thresh=1E8, 
                badpix=None, ipath=None, rclip=None, boff=0, fhot=None, fref=None, autodespike=False, return_mask=False, vrange=None):
    """
        Calculate the system noise temperature Tsys spectrum
        
        Input:
        cpdirnum:    scan number of calibration observation directory
        lin:         line number (dependent on input data level (e.g. 0.6, differ from 0.6.1, 0.6.2, or 0.6.3)
        Tsky:        sky temperature (default: 45 K, provided by Craig)
        exTsys:      expected Tsys, serves as replacement value if observation is corrupted default: 1600 K)
        cleanflag:   if set, cleaning of spikes is performed (default: True)
        thresh:      intensity threshold for cleaning pixels (default: 1E8)
        rep:         cleaning report
        rclip:       replace end of spectrum pixels with neighboring value (a.k.a. remove bad values)
        boff:        number of wing pixels (off spike center) to be corrected (using interpolation)
        fhot:        filename for hot load measurement to be used
        rhot:        filename for ref measurement to be used
        autodespike: automatic despiking of raw data files
        
        Output:
        Tsys:      system noise temperature spectrum
        info:      info about the Tsys scans
        rep:       cleaning report
    """
    badTsys = False    # we hope for a good Tsys
    # get the hot load observation file name
    if fhot==None:
        if ipath!=None: hname = os.path.join(ipath,chdirnum,'*HOT*.fits')
        else: hname = os.path.join('./',chdirnum,'*HOT*.fits')
    else:
        hname = fhot
    hfiles = glob.glob(hname)
    #if verbose: 
    if verbose: print('Cal Hot load: ', hfiles, chdirnum, ipath, hname, vrange)
    
    # read the data
    if verbose: print('hfiles: ', hfiles[0])
    lvv, hot, hotmask, lpos, lh1 = readSTO2LineM(hfiles[0], lin, verbose=verbose, rclip=rclip, badpix=badpix, vrange=vrange)
    #if autodespike: lvvn, hot = removeRawSpike1D(lvv, hot)
    if verbose: print('hot: ', hot.shape)
    
    # repair the data
    #rphrep = mkRep(badpix, lvv, hot, lh1, 'hot')
    #if rphrep!=None: rep = np.vstack((rep,rphrep))
    #if cleanflag: 
    #    if verbose: print('Cleaning hot.')
    #    hot, hrep = cleanSpecPeak1D(hot, thresh=thresh, getReport=True, vv=lvv, boff=boff)
    #    rep = repUpdate(rep, hrep, np.int(lh1['SCAN']), np.float(lh1['OBSID']), 'hot')

    Thot = np.float(lh1['CONELOAD']) * u.K
    if verbose: print('load temperature: %.3f %s'%(Thot.value,Thot.unit))

    # otf cal ref measurement; take the first REF... file, although there should only be one
    if fref==None:
        if ipath!=None: hrname = os.path.join(ipath,chdirnum,'*REF*.fits')
        else: hrname = os.path.join('./',chdirnum,'*REF*.fits')
    else:
        hrname = fref
    hrfiles = glob.glob(hrname)
    if verbose: print('Cal Ref load: ', hrfiles[0])
    
    # read the data
    if verbose: print('hrfiles: ', hrfiles[0])
    hrv, sky, skymask, hrpos, hrh1 = readSTO2LineM(hrfiles[0], lin, verbose=verbose, rclip=rclip, badpix=badpix, vrange=vrange)
    #if autodespike: hrvn, sky = removeRawSpike1D(hrv, sky)
    
    #rprrep = mkRep(badpix, hrv, sky, hrh1, 'load')
    #if rprrep!=None: rep = np.vstack((rep,rprrep))
    #if cleanflag: 
    #    if verbose: print('Cleaning sky.')
    #    hrf, rrep = cleanSpecPeak1D(sky, thresh=thresh, getReport=True, vv=hrv, boff=boff)
    #    rep = repUpdate(rep, rrep, np.int(hrh1['SCAN']), np.float(hrh1['OBSID']), 'sky')
    # if verbose: print('%s %s  %11.6f %10.6f'%(os.path.basename(hfiles[0]), os.path.basename(hrfiles[0]),lpos.galactic.l.degree,lpos.galactic.b.degree))

    y = hot / sky
    if verbose: print(Thot, Tsky, y)
    Tsys = (Thot - y * Tsky) / (y - 1.) 
    if verbose: print('mean Tsys: ', np.mean(Tsys))
    tmask = np.bitwise_or(skymask, hotmask)
    
    # in case of very corrupted Tsys
    if ((np.mean(Tsys.value)<100.)|(np.mean(Tsys.value)>3000.)|(np.isnan(np.nanmean(Tsys.value)))): 
        badTsys = True
        Tsys = np.zeros([hot.shape[0]], dtype=np.float) + exTsys
        
    if Tsys.unit!='K': Tsys.unit = u.K
    
    info = np.array([(badTsys, np.int(lh1['SCAN']), np.float(lh1['OBSID']), np.int(hrh1['SCAN']), np.float(hrh1['OBSID']), Tsky.value, Thot.value)], 
                    dtype=[('badTsys',bool),('hscan', 'i4'),('hobsid', 'i4'),('hrscan', 'i4'),('hrobsid', 'i4'),('Tsky','f8'),('Thot','f8')])
    if return_mask:
        return Tsys, lvv, info, rep, tmask
    else:
        return Tsys, lvv, info, rep




def getTsysLineClean(chdirnum, lin, rep=None, verbose=False, Tsky=45.*u.K, exTsys=1600.*u.K, cleanflag=False, thresh=1E8, 
                badpix=None, ipath=None, rclip=None, boff=0, fhot=None, fref=None, autodespike=False, return_mask=False, vrange=None):
    """
        Calculate the system noise temperature Tsys spectrum
        
        Input:
        cpdirnum:    scan number of calibration observation directory
        lin:         line number (dependent on input data level (e.g. 0.6, differ from 0.6.1, 0.6.2, or 0.6.3)
        Tsky:        sky temperature (default: 45 K, provided by Craig)
        exTsys:      expected Tsys, serves as replacement value if observation is corrupted default: 1600 K)
        cleanflag:   if set, cleaning of spikes is performed (default: True)
        thresh:      intensity threshold for cleaning pixels (default: 1E8)
        rep:         cleaning report
        rclip:       replace end of spectrum pixels with neighboring value (a.k.a. remove bad values)
        boff:        number of wing pixels (off spike center) to be corrected (using interpolation)
        fhot:        filename for hot load measurement to be used
        rhot:        filename for ref measurement to be used
        autodespike: automatic despiking of raw data files
        
        Output:
        Tsys:      system noise temperature spectrum
        info:      info about the Tsys scans
        rep:       cleaning report
    """
    badTsys = False    # we hope for a good Tsys
    # get the hot load observation file name
    if fhot==None:
        if ipath!=None: hname = os.path.join(ipath,chdirnum,'*HOT*.fits')
        else: hname = os.path.join('./',chdirnum,'*HOT*.fits')
    else:
        hname = fhot
    hfiles = glob.glob(hname)
    #if verbose: 
    if verbose: print('Cal Hot load: ', hfiles, chdirnum, ipath, hname, vrange)
    
    # read the data
    if verbose: print('hfiles: ', hfiles[0])
    lvv, hot, hotmask, lpos, lh1 = readSTO2LineM(hfiles[0], lin, verbose=verbose, rclip=rclip, badpix=badpix, vrange=vrange)
    if autodespike: lvvn, hot = removeRawSpike1D(lvv, hot)
    if verbose: print('hot: ', hot.shape)
    
    # repair the data
    rphrep = mkRep(badpix, lvv, hot, lh1, 'hot')
    if rphrep!=None: rep = np.vstack((rep,rphrep))
    if cleanflag: 
        if verbose: print('Cleaning hot.')
        hot, hrep = cleanSpecPeak1D(hot, thresh=thresh, getReport=True, vv=lvv, boff=boff)
        rep = repUpdate(rep, hrep, np.int(lh1['SCAN']), np.float(lh1['OBSID']), 'hot')

    Thot = np.float(lh1['CONELOAD']) * u.K
    if verbose: print('load temperature: %.3f %s'%(Thot.value,Thot.unit))

    # otf cal ref measurement; take the first REF... file, although there should only be one
    if fref==None:
        if ipath!=None: hrname = os.path.join(ipath,chdirnum,'*REF*.fits')
        else: hrname = os.path.join('./',chdirnum,'*REF*.fits')
    else:
        hrname = fref
    hrfiles = glob.glob(hrname)
    if verbose: print('Cal Ref load: ', hrfiles[0])
    
    # read the data
    if verbose: print('hrfiles: ', hrfiles[0])
    hrv, sky, skymask, hrpos, hrh1 = readSTO2LineM(hrfiles[0], lin, verbose=verbose, rclip=rclip, badpix=badpix, vrange=vrange)
    if autodespike: hrvn, sky = removeRawSpike1D(hrv, sky)
    
    rprrep = mkRep(badpix, hrv, sky, hrh1, 'load')
    if rprrep!=None: rep = np.vstack((rep,rprrep))
    if cleanflag: 
        if verbose: print('Cleaning sky.')
        hrf, rrep = cleanSpecPeak1D(sky, thresh=thresh, getReport=True, vv=hrv, boff=boff)
        rep = repUpdate(rep, rrep, np.int(hrh1['SCAN']), np.float(hrh1['OBSID']), 'sky')
    # if verbose: print('%s %s  %11.6f %10.6f'%(os.path.basename(hfiles[0]), os.path.basename(hrfiles[0]),lpos.galactic.l.degree,lpos.galactic.b.degree))

    y = hot / sky
    if verbose: print(Thot, Tsky, y)
    Tsys = (Thot - y * Tsky) / (y - 1.) 
    if verbose: print('mean Tsys: ', np.mean(Tsys))
    tmask = np.bitwise_or(skymask, hotmask)
    
    # in case of very corrupted Tsys
    if ((np.mean(Tsys.value)<100.)|(np.mean(Tsys.value)>3000.)|(np.isnan(np.nanmean(Tsys.value)))): 
        badTsys = True
        Tsys = np.zeros([hot.shape[0]], dtype=np.float) + exTsys
        
    if Tsys.unit!='K': Tsys.unit = u.K
    
    info = np.array([(badTsys, np.int(lh1['SCAN']), np.float(lh1['OBSID']), np.int(hrh1['SCAN']), np.float(hrh1['OBSID']), Tsky.value, Thot.value)], 
                    dtype=[('badTsys',bool),('hscan', 'i4'),('hobsid', 'i4'),('hrscan', 'i4'),('hrobsid', 'i4'),('Tsky','f8'),('Thot','f8')])
    if return_mask:
        return Tsys, lvv, info, rep, tmask
    else:
        return Tsys, lvv, info, rep





def repUpdate(rep, irep, scan, obsid, obstype, verbose=False):
    """
       updating the log (aka rep or report) of cleaned spike pixels
       rep:   existing global report
       irep:  new addendum to report (can be None!)
       scan:  scan number
       obsid: observation ID
       obstype: observation type: hot, load, uref, dref, sig
    """
    if irep!=None:
        irep['scan'] = scan
        irep['obsid'] = obsid
        irep['type'] = obstype
        
        #if rep!=None: rep = np.vstack((rep,irep))
        #if len(rep)>0: rep = np.vstack((rep,irep))
        #else: rep = irep
        if np.type(rep) is np.ndarray: rep = np.vstack((rep,irep))
        else: rep = irep
            
    if verbose: print('repUpdate:  obstype: %s pixels repaired'%(obstype))

    # return the updated (or not) report
    return rep


                
if __name__ == '__main__':
    
    # Careful, the data files might have moved....
    ipath = '/rdat/Projects/'
    
    test = 0
    
    if test ==0:
        ifile = ipath+'STO-2/Data/level0.7/03801/OTF03801_08222.fits'

        vv, spec, pos, hd1, dd1, rs = readSTO2Line(ifile, 2, retdd=True, retcl=True)
        #print(rs.getHistory())
        #print(rs.getComments())
        print(rs.getPos())
        
    if test==1:
        # Examples for Level 0.5 with [CII]: lin = 3
        ifile = ipath+'STO-2/Data/level0.7/03559/OTF03559_00591.fits'
        vv, spec, pos, hd1, dd1 = readSTO2Line(ifile, 2, retdd=True, rclip=2)
        #vv, spec, pos, hd1 = readSTO2(ifile, 3, rclip=2)
        
        # example for Level 0.6 (line numbering has changed: [CII] is now lin = 2)
        ifile = ipath+'STO-2/Data/level0.7/03803/OTF03803_08246.fits'
        badpix2 = [481,482,483,484,485,486,487,488,489]    # bad pixels in cal scans
        
        badpix = None
        vv, spec, pos, hd1, dd1, rs = readSTO2Line(ifile, 2, retdd=True, retcl=True, rclip=2, badpix=badpix)
        print(spec.shape)
        print(vv[badpix2])
        print(spec[badpix2])
        spec = repPix(spec, badpix2)
        print(spec[badpix2])
        print(dir(rs.getHeader()))
        
        sunit = rs.getSpecUnit()
        vunit = rs.getVeloUnit()
        dat   = rs.getDataAsArrays()
        print(dat['velo'].shape)
        print('Version of STO-2 Data read class: ', rs.__version__)
        
        cspec, report = cleanSpecPeak1D(spec,verbose=False, thresh=1E9, getReport=True, vv=vv, bflag=True, boff=1)
        nvv, ncspec = resampleSpectrum(vv, cspec, 2)
        nvv2, ncspec2 = resampleSpectrum(vv, cspec, 2)
        print(nvv[0:20])
        print(nvv2[0:20])
        print(ncspec[0:20])
        print(ncspec2[0:20])
        smspec = smoothData(spec)
        
        ncspec.value[484] = 1.4E7
        ncspec.value[483] = 1.3E7
        ncspec.value[482] = 1.1E7
        
        ncspec = repPix(ncspec, [483,482,484])
        #print(ncspec[badpix])
        
        import matplotlib.pylab as plt
        
        fig = plt.figure()
        sp1 = plt.subplot()
        
        pl1 = plt.plot(vv, spec, color='blue', label='original spectrum')
        #pl1 = plt.plot(vv, spec, '+', color='blue')
        pl2 = plt.plot(nvv, ncspec, color='red', label='cleaned, resampled')
        #pl2 = plt.plot(nvv, ncspec, '+', color='red')
        pl3 = plt.plot(vv, smspec, color='green', label='smoothed spectrum')
        
        plt.xlabel('Velocity (%s)'%(vunit))
        plt.ylabel('Intensity (%s)'%(sunit))
        plt.legend()
        
        #print(report)
        plt.show()

    if test==2:
        # testing repPix1D
        dd = np.arange(2*3*1024).reshape(6,1024)*u.ct
        rpix = [90,91,95]                   # bad test pixels
        dd[...,rpix] = [3,4,5]*u.ct         # set the pixels to "bad" values
        disp = [89,90,91,92,93,94,95,96]    # pixels to be displayed
        print('with bad values: ', dd[...,disp].value)
        nd = repPix(dd, rpix)             # "repair" pixels
        print('values changed: ',nd[...,disp].value)
    
    if test==3:
        Tsys, Tvv, info, rep = getTsysLine('03808', 2, ipath=ipath+'STO-2/Data/level0.7', verbose=True)
        print(info)
        print((np.isinf(np.nanmean(Tsys.value))))
        print(np.isnan(np.nanmean(Tsys.value)))
        print(np.nanmean(Tsys.value))
        Tsys, Tvv, info, rep = getTsysLine('03806', 2, ipath=ipath+'STO-2/Data/level0.7', verbose=True)
        print(info)
        print((np.isinf(np.nanmean(Tsys.value))))
        print(np.isnan(np.nanmean(Tsys.value)))
        print(np.nanmean(Tsys.value))
