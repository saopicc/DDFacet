import sys,math,numpy,os

from DDFacet.Other import MyLogger
log = MyLogger.getLogger("ClassFITSBeam")

import pyrap.measures
import pyrap.quanta
import pyrap.tables

dm = pyrap.measures.measures()
dq = pyrap.quanta

# This a list of the Stokes enums (as defined in casacore header measures/Stokes.h)
# These are referenced by the CORR_TYPE column of the MS POLARIZATION subtable.
# E.g. 5,6,7,8 corresponds to RR,RL,LR,LL
MS_STOKES_ENUMS = [
    "Undefined", "I", "Q", "U", "V", "RR", "RL", "LR", "LL", "XX", "XY", "YX", "YY", "RX", "RY", "LX", "LY", "XR", "XL", "YR", "YL", "PP", "PQ", "QP", "QQ", "RCircular", "LCircular", "Linear", "Ptotal", "Plinear", "PFtotal", "PFlinear", "Pangle"
  ];
# set of circular correlations
CIRCULAR_CORRS = set(["RR", "RL", "LR", "LL"]);
# set of linear correlations
LINEAR_CORRS = set(["XX", "XY", "YX", "YY"]);


class ClassFITSBeam (object):
    def __init__ (self, ms, opts):
        self.ms = ms
        self.filename = opts["FITSFile"]

        # make masure for zenith
        self.zenith = dm.direction('AZEL','0deg','90deg')
        # make position measure from antenna 0
        # NB: in the future we may want to treat position of each antenna separately. For
        # a large enough array, the PA w.r.t. each antenna may change! But for now, use
        # the PA of the first antenna for all calculations
        self.pos0 = dm.position('itrf',*[ dq.quantity(x,'m') for x in self.ms.StationPos[0] ]) 

        # make direction measure from field centre
        self.field_centre = dm.direction('J2000',dq.quantity(self.ms.rarad,"rad"),dq.quantity(self.ms.decrad,"rad"))

        # get channel frequencies from MS
        self.freqs = self.ms.ChanFreq.ravel()

        # NB: need to check correlation names better. This assumes four correlations in that order!
        if "x" in self.ms.CorrelationNames[0]:
            CORRS = "xx","xy","yx","yy"
            print>>log,"polarization basis is linear"
        else:
            CORRS = "rr","rl","lr","ll"
            print>>log,"polarization basis is circular"
        # Following code is nicked from Cattery/Siamese/OMS/pybeams_fits.py
        REIM = "re","im";
        REALIMAG = dict(re="real",im="imag");

        # get the Cattery
        for varname in 'CATTERY_PATH',"MEQTREES_CATTERY_PATH":
            if varname in os.environ:
                sys.path.append(os.environ[varname])

        import Siamese.OMS.Utils as Utils
        import Siamese

        def make_beam_filename (filename_pattern,corr,reim):
            """Makes beam filename for the given correlation and real/imaginary component (one of "re" or "im")"""
            return Utils.substitute_pattern(filename_pattern,
                      corr=corr.lower(),xy=corr.lower(),CORR=corr.upper(),XY=corr.upper(),
                      reim=reim.lower(),REIM=reim.upper(),ReIm=reim.title(),
                      realimag=REALIMAG[reim].lower(),REALIMAG=REALIMAG[reim].upper(),
                      RealImag=REALIMAG[reim].title());

        filename_real = [];
        filename_imag = [];
        for corr in CORRS:
            # make FITS images or nulls for real and imaginary part
            filename_real.append(make_beam_filename(self.filename,corr,'re'))
            filename_imag.append(make_beam_filename(self.filename,corr,'im'))

        # load beam interpolator
        print 'Loading FITS Beams'
        import Siamese.OMS.InterpolatedBeams as InterpolatedBeams
        self.vbs = []
        for reFits, imFits in zip(filename_real,filename_imag):        
            print>>log,"Loading beam patterns",filename_real,filename_imag
            vb = InterpolatedBeams.LMVoltageBeam(verbose=0,l_axis="-X",m_axis="Y")  # verbose, XY must come from options
            vb.read(reFits,imFits)
            self.vbs.append(vb)

    def getFreqs (self):
        return self.freqs

    def evaluateBeam (self, t0, ra, dec):
        """Evaluates beam at time t0, in directions ra, dec.
        Inputs: t0 is a single time. ra, dec are Ndir vectors of directions.
        Output: a complex array of shape [Ndir,Nant,Nfreq,2,2] giving the Jones matrix per antenna, direction and frequency
        """

        # put antenna0 position as reference frame. NB: in the future may want to do it per antenna
        dm.do_frame(self.pos0);
        # put time into reference frame
        dm.do_frame(dm.epoch("UTC",dq.quantity(t0,"s")))
        # compute PA 
        parad = dm.posangle(self.field_centre,self.zenith).get_value("rad")
        # print>>log,"time %f, position angle %f"%(t0, parad*180/math.pi)

        # compute l,m per direction
        ndir = len(ra)
        l0 = numpy.zeros(ndir,float)
        m0 = numpy.zeros(ndir,float)
        for i,(r1,d1) in enumerate(zip(ra,dec)):
          l0[i], m0[i] = self.ms.radec2lm_scalar(r1,d1)

        # rotate each by parallactic angle
        r = numpy.sqrt(l0*l0+m0*m0)
        angle = numpy.arctan2(m0,l0)
        l = r*numpy.cos(angle+parad)
        m = r*numpy.sin(angle+parad)  

        # get interpolated values. Output shape will be [ndir,nfreq]
        beamjones = [ self.vbs[i].interpolate(l,m,freq=self.freqs,freqaxis=1) for i in range(4) ]

        # now make output matrix
        jones = numpy.zeros((ndir,self.ms.na,len(self.freqs),2,2),dtype=numpy.complex64)

        # populate it with values
        # NB: here we copy the same Jones to every antenna. In principle we could compute
        # a parangle per antenna. When we have pointing error, it's also going to be per
        # antenna
        for iant in xrange(self.ms.na):
            for ijones,(ix,iy) in enumerate(((0,0),(0,1),(1,0),(1,1))):
                jones[:,iant,:,ix,iy] = beamjones[ijones]
        return jones







