import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassStokes")
import re

'''
Enumeration of stokes and correlations used in MS2.0 - as per Stokes.h in casacore, the rest is left unimplemented:
'''
StokesTypes = {'I': 1, 'Q': 2, 'U': 3, 'V': 4, 'RR': 5, 'RL': 6, 'LR': 7, 'LL': 8, 'XX': 9, 'XY': 10, 'YX': 11,
               'YY': 12}
'''
See Smirnov I (2011) for description on conversion between correlation terms and stokes params for linearly polarized feeds
See Synthesis Imaging II (1999) Pg. 9 for a description on conversion between correlation terms and stokes params for circularly polarized feeds

Here the last value in each of the options specifies how to combine the individual terms
to form the parameter. Currently we support (dep_index1) or contant(dep_index1 op dep_index2) or
"constant"i(dep_index1 op dep_index2)
'''

StokesDependencies = {
    'I'  : [[StokesTypes['I'],"(0)"], [StokesTypes['RR'], StokesTypes['LL'],"0.5(0+1)"],
            [StokesTypes['XX'], StokesTypes['YY'],"0.5(0+1)"]],
    'V'  : [[StokesTypes['V'],"(0)"], [StokesTypes['RR'], StokesTypes['LL'],"0.5(0-1)"],
            [StokesTypes['XY'], StokesTypes['YX'],"-0.5i(0-1)"]],
    'U'  : [[StokesTypes['U'],"(0)"], [StokesTypes['RL'], StokesTypes['LR'],"-0.5i(0-1)"],
            [StokesTypes['XY'], StokesTypes['YX'],"0.5(0+1)"]],
    'Q'  : [[StokesTypes['Q'],"(0)"], [StokesTypes['RL'], StokesTypes['LR'],"0.5(0+1)"],
            [StokesTypes['XX'], StokesTypes['YY'],"0.5(0-1)"]],
    'RR' : [[StokesTypes['RR'],"(0)"]],
    'RL' : [[StokesTypes['RL'],"(0)"]],
    'LR' : [[StokesTypes['LR'],"(0)"]],
    'LL' : [[StokesTypes['LL'],"(0)"]],
    'XX' : [[StokesTypes['XX'],"(0)"]],
    'XY' : [[StokesTypes['XY'],"(0)"]],
    'YX' : [[StokesTypes['YX'],"(0)"]],
    'YY' : [[StokesTypes['YY'],"(0)"]]
}

class ClassStokes:
    """
    Converts between correlations (predominantly used visibility space) and Stokes
    parameters (used in image space)
    Args:
        MSDataDescriptor:
        FITSPolDescriptor:
    """
    def __init__(self, MSDataDescriptor, FITSPolDescriptor):
        self._MSDataDescriptor = MSDataDescriptor
        self._MScorrLabels = []
        self._gridMSCorrMapping = {}
        for corrId, corr in enumerate(self._MSDataDescriptor):
            if corr not in StokesTypes.values():
                raise ValueError("Measurement Set contains unsupported correlation specifier %d" % corr)
            for k in StokesTypes.keys():
                if corr == StokesTypes[k]:
                    self._MScorrLabels.append(k)
                    self._gridMSCorrMapping[corr] = corrId

        print >> log, "Correlations stored in Measurement set is labeled as %s" % self._MScorrLabels

        self._FITSstokesList = []
        if type(FITSPolDescriptor) is list:
            for corr in FITSPolDescriptor:
                if corr not in StokesTypes:
                    raise ValueError("Polarization %s invalid must be one of %s" % (corr, StokesTypes.keys()))
                self._FITSstokesList.append(corr)
        elif type(FITSPolDescriptor) is str:
            if not re.match(r"^[IQUV]{1,4}$", FITSPolDescriptor.strip()):
                raise ValueError("Polarization stokes string invalid must be one or more of IQUV")
            for stokes in FITSPolDescriptor.strip():
                self._FITSstokesList.append(stokes)
        else:
            raise ValueError("Image polarization descriptor must be stokes string or string list")
        self._stokesExpr = []
        for stokesId, stokes in enumerate(self._FITSstokesList):
            depOptions = StokesDependencies[stokes]
            satisfied = False
            for dep_i,dep in enumerate(depOptions):
                depIndicies = dep[0:len(dep)-1] #skip combining expression
                if False not in [(corr in self._MSDataDescriptor) for corr in depIndicies]:
                    satisfied = True
                    self._stokesExpr.append(dep_i)
                    break
            if not satisfied:
                raise ValueError("Required data for stokes term %s not "
                                 "available in MS. Verify your parset options." % stokes)

        print >> log, "Stokes parameters required in FITS cube: %s" % self._FITSstokesList

    @staticmethod
    def _extractStokesCombinationExpression(exp):
        """

        Args:
            exp:

        Returns:

        """
        if type(exp) is not list or len(exp) < 2:
            raise ValueError("Expected stokes dependency of the form x0 [x1 ... xN] expr")
        deps = exp[0:len(exp)-1]
        combExp = exp[len(exp)-1]
        vals = re.match(r"(?P<scalar>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)?(?P<imag>i)?"
                        r"\((?P<firstId>[0-9])(?:(?P<op>[+-])(?P<secondId>[0-9]))?\)",combExp)

        if vals is None:
            raise ValueError("Expression for combination of stokes parameters malformed")
        sc = 1 if vals.group("scalar") is None else float(vals.group("scalar"))
        imag = 1 if vals.group("imag") is None else 0+1j
        firstId = exp[int(vals.group("firstId"))]

        supportedOps = {"+":np.add, "-":np.subtract, None:(lambda x,y: x)}
        op = supportedOps[vals.group("op")]
        secondId = 1 if vals.group("secondId") is None else exp[int(vals.group("secondId"))]
        return (sc,imag,firstId,op,secondId)

    def corrsToStokes(self, corrCube):
        """
        Args:
            corrCube
        Returns:
            Image cube with stokes parametres as specified by initializer
        """
        nChan = corrCube.shape[0]
        nCorrIn = corrCube.shape[1]
        nStokesOut = len(self._FITSstokesList)
        nV = corrCube.shape[2]
        nU = corrCube.shape[3]
        stokesCube = np.empty([nChan,nStokesOut,nV,nU], dtype=corrCube.dtype)
        for stokesId, (stokes, depExpr) in enumerate(zip(self._FITSstokesList, self._stokesExpr)):
            ops = self._extractStokesCombinationExpression(StokesDependencies[stokes][depExpr])
            np.add(corrCube[:, self._gridMSCorrMapping[ops[2]], :, :],corrCube[:, self._gridMSCorrMapping[ops[4]], :, :])
            stokesCube[:, stokesId, :, :] = (ops[0]*ops[1])*ops[3](corrCube[:, self._gridMSCorrMapping[ops[2]], :, :],
                                                                   corrCube[:, self._gridMSCorrMapping[ops[4]], :, :])
        return stokesCube