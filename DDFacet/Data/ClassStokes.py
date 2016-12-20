'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

import re

import numpy as np
from DDFacet.Other import MyLogger

log= MyLogger.getLogger("ClassStokes")

'''
Enumeration of stokes and correlations used in MS2.0 - as per Stokes.h in casacore, the rest are left unimplemented:

These are useful when working with visibility data (https://casa.nrao.edu/Memos/229.html#SECTION000613000000000000000)
'''
StokesTypes = {'I': 1, 'Q': 2, 'U': 3, 'V': 4, 'RR': 5, 'RL': 6, 'LR': 7, 'LL': 8, 'XX': 9, 'XY': 10, 'YX': 11,
               'YY': 12}

'''
The following definition can be found in Table 28,
Definition of the Flexible Image Transport System (FITS),   version 3.0
W. D.  Pence, L.  Chiappetti, C. G.  Page, R. A.  Shaw, E.  Stobie
A&A 524 A42 (2010)
DOI: 10.1051/0004-6361/201015362

(These are useful only when writing out to FITS files)
'''
FitsStokesTypes = {
    "I" : 1, #Standard Stokes unpolarized
    "Q" : 2, #Standard Stokes linear
    "U" : 3, #Standard Stokes linear
    "V" : 4, #Standard Stokes circular
    "RR": -1, #Right-right circular
    "LL": -2, #Left-left circular
    "RL": -3, #Right-left cross-circular
    "LR": -4, #Left-right cross-circular
    "XX": -5, #X parallel linear
    "YY": -6, #Y parallel linear
    "XY": -7, #XY cross linear
    "YX": -8  #YX cross linear
}

'''
See Smirnov I (2011) for description on conversion between correlation terms and stokes params for linearly polarized feeds
See Synthesis Imaging II (1999) Pg. 9 for a description on conversion between correlation terms and stokes params for circularly polarized feeds

Here the last value in each of the options specifies how to combine the individual terms
to form the parameter. See _extract for documentation on the accepted conversion expressions
specified as last parameter of each of the dependencies in the listings.
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

'''
Below are the expressions for the reverse transformations from Stokes to correlations:
'''
CorrelationDependencies = {
    'XX' : [[StokesTypes["XX"], "(0)"], [StokesTypes["I"], StokesTypes["Q"], "(0+1)"]],
    'XY' : [[StokesTypes["XY"], "(0)"], [StokesTypes["U"], StokesTypes["V"], "(0+i1)"]],
    'YX' : [[StokesTypes["YX"], "(0)"], [StokesTypes["U"], StokesTypes["V"], "(0-i1)"]],
    'YY' : [[StokesTypes["YY"], "(0)"], [StokesTypes["I"], StokesTypes["Q"], "(0-1)"]],
    'RR' : [[StokesTypes["RR"], "(0)"], [StokesTypes["I"], StokesTypes["V"], "(0+1)"]],
    'RL' : [[StokesTypes["RL"], "(0)"], [StokesTypes["Q"], StokesTypes["U"], "(0+i1)"]],
    'LR' : [[StokesTypes["LR"], "(0)"], [StokesTypes["Q"], StokesTypes["U"], "(0-i1)"]],
    'LL' : [[StokesTypes["LL"], "(0)"], [StokesTypes["I"], StokesTypes["V"], "(0-1)"]],
    'I'  : [[StokesTypes["I"], "(0)"]],
    'Q'  : [[StokesTypes["Q"], "(0)"]],
    'U'  : [[StokesTypes["U"], "(0)"]],
    'V'  : [[StokesTypes["V"], "(0)"]]
}
class ClassStokes:
    """
    Converts between correlations (predominantly used in visibility space) and Stokes
    parameters (used in image space). This transformation can be done before or after Fourier inversion (Fourier transform
    preserves addition).
    Args:
        MSDataDescriptor: Correlation descriptor taken from POLARIZATION MS2.0 table, CORR_TYPE field
        FITSPolDescriptor: Descriptor for required polarizations in FITS files (can be IQUV string or list of
                           correlations (and/or polarizations)
    Raises:
        ValueError if the Measurement set doesn't contain supported constants (see StokesTypes)
        ValueError if one of requested polarizations are not supported
        ValueError if requested polarizations is not a stokes string or list
    Post conditions:
        self._MSDataDescriptor are the correlations specified as per input arg (as defined in Stokes.h in casacore)
        self._MScorrLabels are the string identifiers of the input MS correlations
        self._gridMSCorrMapping is the mapping between the Stokes.h correlation identifiers and the index in the visibility
                                array (or grid cube)
        self._FITSstokesList are the Stokes parameters passed to the initializer
        self._FITSstokesSliceLookup is a mapping between the Stokes.h identifier of the polarization and the FITS
                                polarization slice.
        self._stokesExpr is the correlation to stokes dependency resolution (as defined in StokesDependencies) for
                                converting gridded visibilities to gridded stokes, for each required stokes component
        self._corrExpr is the stokes to correlation dependency resolution (as defined in CorrelationDependencies) for
                                converting gridded stokes to gridded visibilities, for each correlation in the measurement
                                CORR_TYPE field
    """
    def __init__(self, MSDataDescriptor, FITSPolDescriptor):
        self._MSDataDescriptor = MSDataDescriptor
        self._MScorrLabels = []
        self._gridMSCorrMapping = {}

        #Check if the correlations specified in the MS are supported by the conversion code:
        for corrId, corr in enumerate(self._MSDataDescriptor):
            if corr not in StokesTypes.values():
                raise ValueError("Measurement Set contains unsupported correlation specifier %s" % corr)
            for k in StokesTypes.keys():
                if corr == StokesTypes[k]:
                    self._MScorrLabels.append(k)
                    self._gridMSCorrMapping[corr] = corrId
        print >> log, "Correlations stored in Measurement set (and gridded) is labeled as %s" % self._MScorrLabels

        #Check if the requested FITS polarizations are supported by the converter:
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
        self._FITSstokesSliceLookup = {}
        for slice, stokes in enumerate(self._FITSstokesList):
            self._FITSstokesSliceLookup[StokesTypes[stokes]] = slice
        print >> log, "Stokes parameters required in FITS cube: %s" % self._FITSstokesList

    @staticmethod
    def _extractStokesCombinationExpression(exp):
        """
        Extracts an expression for computing a stokes or correlation term from its dependencies.


        Args:
            exp: 1 or 2 dependencies (Stokes.h polarization identifiers) for computing either a Stokes or
                 Correlation term, AND a string expression on how to combine dependency 0 and dependency 1.
                 Currently we support (dep_index1) or contant(dep_index1 op dep_index2) or
                 "constant"i(dep_index1 op dep_index2) or (i"dep_index1") or contant(i"dep_index1" op dep_index2) or
                 contant(dep_index1 op i"dep_index2") or "constant"i(i"dep_index1" op dep_index2) or
                 "constant"i(dep_index1 op i"dep_index2")
        Returns: Tuple of:
                 scalar constant to multiply both dependencies by
                 scalar complex constant to multiply both dependencies by
                 index of dependency (0 or 1) to be used as left-most value in the expressions listed above
                 binary operator, currently supports + and - (or left blank the operator is defined as id(left-most dependency)
                 index of dependency (0 or 1) to be used as right-most value in the expressions listed above (compulsory
                        if binary operator was specified)
        """
        if type(exp) is not list or len(exp) < 2:
            raise ValueError("Expected stokes dependency of the form x0 [x1 ... xN] expr")
        deps = exp[0:len(exp)-1]
        combExp = exp[len(exp)-1]
        vals = re.match(r"(?P<scalar>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)?(?P<imag>i)?"
                        r"\((?P<firstIdImag>i)?(?P<firstId>[0-9])(?:(?P<op>[+-])(?P<secondIdImag>i)?(?P<secondId>[0-9]))?\)",combExp)

        if vals is None:
            raise ValueError("Expression for combination of stokes parameters malformed")
        sc = 1 if vals.group("scalar") is None else float(vals.group("scalar"))
        imag = 1 if vals.group("imag") is None else 0+1j
        firstId = exp[int(vals.group("firstId"))]
        supportedOps = {"+":np.add, "-":np.subtract, None:(lambda x,y: x)}
        op = supportedOps[vals.group("op")]
        secondId = 1 if vals.group("secondId") is None else exp[int(vals.group("secondId"))]
        firstIdImag = 1 if vals.group("firstIdImag") is None else 0 + 1j
        secondIdImag = 1 if vals.group("secondIdImag") is None else 0 + 1j
        return (sc,imag,firstId,op,secondId, firstIdImag, secondIdImag)

    def corrs2stokes(self, corrCube):
        """
        Converts a cube of correlations to a cube of stokes parameters
        Args:
            corrCube: numpy complex cube of the form [channels,corrs,nY,nX]
        Raises:
            TypeError if correlation cube is not of complex type (required to compute V from linear feeds and U from
            circular feeds)
            ValueError if one or more stokes products cannot be formed from the correlations present in the MS
        Returns:
            Cube with stokes parameters as specified by initializer
        """

        # Check that the Measurement Set has the required correlation data to form the requested polarization products:
        _stokesExpr = []
        for stokesId, stokes in enumerate(self._FITSstokesList):
            depOptions = StokesDependencies[stokes]
            satisfied = False
            for dep_i, dep in enumerate(depOptions):
                depIndicies = dep[0:len(dep) - 1]  # skip combining expression
                if False not in [(corr in self._MSDataDescriptor) for corr in depIndicies]:
                    satisfied = True
                    _stokesExpr.append(dep_i)
                    break
            if not satisfied:
                raise ValueError("Required data for stokes term %s not "
                                 "available in MS. Verify your parset options." % stokes)

        if not np.iscomplexobj(corrCube):
            raise TypeError("Correlation cube must be of type complex (certain stokes terms cannot be reconstructed from real data)")
        nChan = corrCube.shape[0]
        nCorrIn = corrCube.shape[1]
        nStokesOut = len(self._FITSstokesList)
        nV = corrCube.shape[2]
        nU = corrCube.shape[3]
        stokesCube = np.empty([nChan,nStokesOut,nV,nU], dtype=corrCube.dtype)
        for stokesId, (stokes, depExprId) in enumerate(zip(self._FITSstokesList, _stokesExpr)):
            ops = self._extractStokesCombinationExpression(StokesDependencies[stokes][depExprId])
            stokesCube[:, stokesId, :, :] = (ops[0]*ops[1])*ops[3](ops[5] * corrCube[:, self._gridMSCorrMapping[ops[2]], :, :],
                                                                   ops[6] * corrCube[:, self._gridMSCorrMapping[ops[4]], :, :])
        return stokesCube

    def stokes2corrs(self, stokesCube):
        """
        Converts a cube of stokes components to a cube of correlations
        Args:
            stokesCube: numpy complex cube of the form [channels,stokes,nY,nX]
        Raises:
            TypeError if stokes cube is not of complex type
            ValueError if one or more correlation products cannot be formed from the requested stokes products in the image
        Returns:
            Cube with correlation parameters as specified by initializer
        """

        # Find the Stokes to correlation conversion expressions:
        _corrsExpr = []
        for corrId, corr in enumerate(self._MScorrLabels):
            depOptions = CorrelationDependencies[corr]
            satisfied = False
            for dep_i, dep in enumerate(depOptions):
                depIndicies = dep[0:len(dep) - 1]  # skip combining expression
                if False not in [(stokePar in [StokesTypes[s] for s in self._FITSstokesList])
                                 for stokePar in depIndicies]:
                    satisfied = True
                    _corrsExpr.append(dep_i)
                    break
            if not satisfied:
                raise ValueError("Required data for computing correlation term %s not "
                                 "available in requested imaging Stokes products. Verify your parset options." % corr)

        if not np.iscomplexobj(stokesCube):
            raise TypeError("Stokes cube must be of type complex (certain correlation terms cannot be reconstructed from real data)")
        nChan = stokesCube.shape[0]
        nStokesIn = stokesCube.shape[1]
        nCorrOut = len(self._MSDataDescriptor)
        nV = stokesCube.shape[2]
        nU = stokesCube.shape[3]
        corrCube = np.empty([nChan, nCorrOut, nV, nU], dtype=stokesCube.dtype)
        for corrId, (corr,depExprId) in enumerate(zip(self._MScorrLabels, _corrsExpr)):
            ops = self._extractStokesCombinationExpression(CorrelationDependencies[corr][depExprId])
            corrCube[:,corrId,:,:] = (ops[0]*ops[1])*ops[3](ops[5] * stokesCube[:, self._FITSstokesSliceLookup[ops[2]], :, :],
                                                            ops[6] * stokesCube[:, self._FITSstokesSliceLookup[ops[4]], :, :])

        return corrCube

    def NStokesInImage(self):
        """
        Returns: The number of stokes parameters / correlations in image
        """
        return len(self._FITSstokesList)

    def RequiredStokesProducts(self):
        """
        Returns: Required output stokes parameters
        """
        return self._FITSstokesList

    def RequiredStokesProductsIds(self):
        """
        Returns the stokes.h ids for the required stokes products
        """
        return [StokesTypes[s] for s in self._FITSstokesList]

    def AvailableCorrelationProductsIds(self):
        """
        Returns the stokes.h ids for the available MS correlation products
        """
        return self._MSDataDescriptor