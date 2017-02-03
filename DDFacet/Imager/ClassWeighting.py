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

import math

import DDFacet.cbuild.Gridder._pyGridder as _pyGridder
import DDFacet.cbuild.Gridder._pyWeightingCore as WeightingCore
import numpy as np
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor

log= MyLogger.getLogger("ClassWeighting")


#import ImagingWeights
from DDFacet.Data import ClassMS
from pyrap.tables import table
from DDFacet.Array import NpShared

def test(field=0,weight="Uniform"):
    print>>log,"reading test MS"
#    MS=ClassMS.ClassMS("/media/6B5E-87D0/killMS2/TEST/Simul/0000.MS")
    MS= ClassMS.ClassMS("CYG-B-test.MS", DoReadData=False)
    t=table(MS.MSName,ack=False).query("FIELD_ID==%d"%field)
    WEIGHT=t.getcol("WEIGHT_SPECTRUM")
    flag_all=t.getcol("FLAG")
    uvw=t.getcol("UVW")
    t.close()
    ImShape=(1, 1, 6125, 6125)
    CellSizeRad=(0.06/3600)*np.pi/180
    CW=ClassWeighting(ImShape,CellSizeRad)
    #CW.CalcWeights(MS.uvw[199:200],WEIGHT[199:200,0:3],MS.flag_all[199:200,0:3],MS.ChanFreq[0:3],Weighting="Uniform")


    flag_all.fill(0)

    # for i in [206]:#range(200,211):
    #     r0,r1=i,i+10
    #     print r0,r1
    #     uvw=np.float64(MS.uvw[r0:r1].copy())
    #     flags=MS.flag_all[r0:r1,0:3].copy()
    #     W=WEIGHT[r0:r1,0:3].copy()
    #     W.fill(1)
    #     freqs=MS.ChanFreq[0:3].copy()
    #     CW.CalcWeights(uvw,W,flags,freqs,Weighting="Uniform")

    WEIGHT = WEIGHT.mean(axis=2)
    WEIGHT.fill(1)
    #flag_all[MS.A0==MS.A1]=1
    #WEIGHT[MS.flag_all[:,:,0]==1]=0
    print>>log,"calculating test weights for shape %s"%(flag_all.shape,)
    CW.CalcWeights(uvw,WEIGHT,flag_all,MS.ChanFreq,Robust=0,Weighting=weight)
    

_cc = 299792458

class ClassWeighting():
    def __init__(self,ImShape,CellSizeRad):
        self.ImShape=ImShape
        self.CellSizeRad=CellSizeRad

    def CalcWeights(self):
        """
        Calculates visibility weights. This can be run in a main or background process.
        """
        if self.VisWeights is not None:
            return

        # in RAM-greedy mode, we keep all weight arrays around in RAM while computing weights
        # Otherwise we leave them in mmap()ed files and etach, and let the
        # kernel take care of caching etc.
        greedy = self.GD["Debug"]["MemoryGreedy"]

        # check if every MS+chunk weight is available in cache
        self.VisWeights = shared_dict.create("VisWeights")
        have_all_weights = True
        for iMS, MS in enumerate(self.ListMS):
            msweights = self.VisWeights.addSubdict(iMS)
            for iChunk, (row0, row1) in enumerate(MS.getChunkRow0Row1()):
                cachepath, valid = MS.getChunkCache(row0, row1).checkCache("ImagingWeights.npy",
                                                                           dict([(section, self.GD[section]) for section
                                                                                 in ("Data", "Selection", "Freq",
                                                                                     "Image")]))
                have_all_weights = have_all_weights and valid
                w = msweights.addSubdict(iChunk)
                w['path'] = cachepath
        # if every weight is in cache, then VisWeights has been constructed properly -- return, else
        # carry on to compute it
        if have_all_weights:
            print>> log, "all imaging weights are available in cache"
            return

        # Read UVs, weights, flags, freqs from all MSs in the list.
        # Form up output_list of (uv,weights,flags,freqs) tuples, one per each MS in self.ListMS, where shapes are
        # (nrow,2), (nrow,nchan), (nrow) and (nchan) respectively.
        # Note that flags are "row flags" i.e. only True if all channels are flagged.
        # Per-channel flagging is taken care of in here, by setting that
        # channel's weight to 0.

        WeightCol = self.GD["Weight"]["ColName"]
        # now loop over MSs and read data
        weightsum = nweights = 0
        weights_are_null = True
        wmax = 0
        uvmax_wavelengths = 0
        for iMS, ms in enumerate(VS.ListMS):
            msweights = self.VisWeights[iMS]
            msfreqs = ms.ChanFreq
            tab = ms.GiveMainTable()
            for iChunk, (row0, row1) in enumerate(ms.getChunkRow0Row1()):
                msw = msweights[iChunk]
                nrows = row1 - row0
                chanslice = ms.ChanSlice
                if not nrows:
                    print>> log, "  0 rows: empty chunk"
                    continue
                print>> log, "  %d.%d reading %s UVW" % (iMS + 1, iChunk + 1, ms.MSName)
                uvw = tab.getcol("UVW", row0, nrows)
                flags = np.empty((nrows, len(ms.ChanFreq), len(ms.CorrelationIds)), np.bool)
                # print>>log,(ms.cs_tlc,ms.cs_brc,ms.cs_inc,flags.shape)
                print>> log, "  reading FLAG" % ms.MSName
                tab.getcolslicenp("FLAG", flags, ms.cs_tlc, ms.cs_brc, ms.cs_inc, row0, nrows)
                if ms._reverse_channel_order:
                    flags = flags[:, ::-1, :]
                # if any polarization is flagged, flag all 4 correlations. Shape of flags becomes nrow,nchan
                print>> log, "  adjusting flags"
                # if any polarization is flagged, flag all 4 correlations. Shape
                # of flags becomes nrow,nchan
                flags = flags.max(axis=2)
                # if all channels are flagged, flag whole row. Shape of flags becomes nrow
                rowflags = flags.min(axis=1)
                # if everything is flagged, skip this entry
                if rowflags.all():
                    print>> log, "  all flagged: marking as null"
                    continue
                # if all channels are flagged, flag whole row. Shape of flags becomes nrow
                msw["flags"] = rowflags
                # adjust max uv (in wavelengths) and max w
                wmax = max(uvw[:, 3].max(), wmax)
                msw["uv"] = uvw[:2]
                del uvw
                # max of |u|, |v| in wavelengths
                uvm = abs(msw["uv"][~rowflags, :]).max() * msfreqs.max() / _cc
                uvmax_wavelengths = max(uvmax_wavelengths, uvm)
                # now read the weights
                weight = msw.addSharedArray("weight", (nrows, ms.Nchan), np.float32)
                if WeightCol == "WEIGHT_SPECTRUM":
                    w = tab.getcol(WeightCol, row0, nrows)[:, chanslice]
                    print>> log, "  reading column %s for the weights, shape is %s" % (
                        WeightCol, w.shape)
                    if ms._reverse_channel_order:
                        w = w[:, ::-1, :]
                    # take mean weight across correlations and apply this to all
                    weight[...] = w.mean(axis=2)
                elif WeightCol == "None" or WeightCol == None:
                    print>> log, "  Selected weights columns is None, filling weights with ones"
                    weight.fill(1)
                elif WeightCol == "WEIGHT":
                    w = tab.getcol(WeightCol, row0, nrows)
                    print>> log, "  reading column %s for the weights, shape is %s, will expand frequency axis" % (
                        WeightCol, w.shape)
                    # take mean weight across correlations, and expand to have frequency axis
                    weight[...] = w.mean(axis=1)[:, np.newaxis]
                else:
                    # in all other cases (i.e. IMAGING_WEIGHT) assume a column
                    # of shape NRow,NFreq to begin with, check for this:
                    w = tab.getcol(WeightCol, row0, nrows)[:, chanslice]
                    print>> log, "  reading column %s for the weights, shape is %s" % (
                        WeightCol, w.shape)
                    if w.shape != valid.shape:
                        raise TypeError(
                            "weights column expected to have shape of %s" %
                            (valid.shape,))
                    weight[...] = w
                # flagged points get zero weight
                weight *= valid
                nullweight = (WEIGHT == 0).all()
                if nullweight:
                    msw.delete_item("weight")
                    msw.delete_item("uv")
                    msw.delete_item("flags")
                else:
                    weights_are_null = False
            tab.close()

        # compute normalization factor
        weightnorm = nweights / weightsum if weightsum else 1
        print>> log, "weight norm is %g (sum %g from %d valid visibility points)" % (
            weightnorm, weightsum, nweights)

        # now compute actual imaging weights
        ImShape = self.FullImShape  # self.FacetShape
        CellSizeRad = self.CellSizeRad
        WeightMachine = ClassWeighting.ClassWeighting(ImShape, CellSizeRad)
        Robust = self.Robust

        if self.MFSWeighting or self.NFreqBands < 2:
            band_mapping = None
        else:
            # we need provide a band mapping for every chunk of weights, so construct a list
            # where each MS's mapping is repeated Nchunk times
            band_mapping = []
            for i, ms in enumerate(self.ListMS):
                band_mapping += [self.DicoMSChanMapping[i]] * ms.Nchunk

        WeightMachine.CalcWeights(
            output_list,
            Robust=Robust,
            Weighting=self.Weighting,
            Super=self.Super,
            nbands=self.NFreqBands if band_mapping is not None else 1,
            band_mapping=band_mapping,
            weightnorm=weightnorm,
            force_unity_weight=weights_are_null)

        # # done, every weight array in output_list has been normalized to proper imaging weights
        # # we now release the arrays, which will flush the buffers to disk
        # # (eventually)
        # del output_list
        # save, and mark the cache as safe
        for iMS, MS in enumerate(self.ListMS):
            for (row0, row1), path in zip(MS.getChunkRow0Row1(), self.VisWeights[iMS]):
                array = weight_arrays.get(path)
                if array is not None:
                    print>> log, "saving %s" % path
                    np.save(path, array)
                else:
                    print>> log, "saving empty %s (all flagged or null)" % path
                    # make an empty weights file in the cache
                    file(path, 'w').truncate(0)
                MS.getChunkCache(row0, row1).saveCache("ImagingWeights.npy")

    def CalcWeights(self, uvw_weights_flags_freqs, Robust=0, Weighting="Briggs", Super=1,
                          nbands=1, band_mapping=None, weightnorm=1, force_unity_weight=False):
        """
        Computes imaging weights in "MFS mode", when all uv-points are binned onto a single grid.
        Args:
            uvw_weights_flags_freqs: list of (uv, weights, flags, freqs) tuples, one per each MS
                if weights is a string, it is treated as the filename for a shared array
            Robust:                  robustness
            Weighting:               natural, uniform, briggs
            Super:                   !=1 for superuniform or superrobust: uv bin size is 1/(super*FoV)
            nbands:                  number of frequency bands to compute weights on (if band_mapping is not None)
            band_mapping:            band_mapping[iMS][ichan] gives the band number of channel #ichan of MS #iMS
                                     if None, the "MFS weighting" is used, with all frequency points weighted
                                     on a single grid
            weightnorm:              multiply weights by this factor
            force_unity_weight:      force all weights to 1

        Returns:
            list of imaging weights arrays, one per MS, same shape as original data weights
        """

        Weighting = Weighting.lower()
        if Weighting == "natural":
            print>> log, "Weighting in natural mode"
            if force_unity_weight:
                for uv, weights, flags, freqs in uvw_weights_flags_freqs:
                    if flags is not None:
                        if type(weights) is str:
                            NpShared.GiveArray(weights).fill(1)
                        else:
                            weights.fill(1)
            return [x[1] for x in uvw_weights_flags_freqs]

        nch, npol, npixIm, _ = self.ImShape
        FOV = self.CellSizeRad * npixIm
        cell = 1. / (Super * FOV)

        if band_mapping is None:
            nbands = 1
            print>> log, "initializing weighting grid for single band (or MFS weighting)"
        else:
            print>> log, "initializing weighting grids for %d bands"%nbands

        # find max grid extent by considering _unflagged_ UVs
        xymax = 0
        for uv, weights, flags, freqs in uvw_weights_flags_freqs:
            # max |u|,|v| in lambda
            uvsel=abs(uv)[~flags, :]
            if uvsel.size==0:
                print>> log, ModColor.Str("  A dataset is fully flagged")
                continue
            uvmax = uvsel.max() * freqs.max() / _cc
            xymax = max(xymax, int(math.floor(uvmax / cell)))
            if flags is not None:
                # max |u|,|v| in lambda
                uvmax = abs(uv)[~flags, :].max() * freqs.max() / _cc
                xymax = max(xymax, int(math.floor(uvmax / cell)))
        if xymax == 0:
            raise Exception('All datasets are fully flagged')

        xymax += 1
        # grid will be from [-xymax,xymax] in U and [0,xymax] in V
        npixx = xymax * 2 + 1
        npixy = xymax + 1
        npix = npixx * npixy


        print>> log, "Calculating imaging weights on an [%i,%i]x%i grid with cellsize %g" % (npixx, npixy, nbands, cell)
        grid0 = np.zeros((nbands, npix), np.float64)
        grid = grid0.reshape((nbands*npix,))

        # this will ve a per-MS list of weights and an index array, or None if an MS is all flagged
        weights_index = [(None, None)] * len(uvw_weights_flags_freqs)

        for iMS, (uv, weights_or_path, flags, freqs) in enumerate(uvw_weights_flags_freqs):
            if flags is None:  # entire chunk flagged
                continue
            weights = NpShared.GiveArray(weights_or_path) if type(weights_or_path) is str else weights_or_path
            if force_unity_weight:
                weights.fill(1)
                weights[flags,...]=0

            elif weightnorm != 1:
                weights *= weightnorm
            # flip sign of negative v values -- we'll only grid the top half of the plane
            uv[uv[:, 1] < 0] *= -1
            # convert u/v to lambda, and then to pixel offset
            uv = uv[..., np.newaxis] * freqs[np.newaxis, np.newaxis, :] / _cc
            uv = np.floor(uv / cell).astype(int)
            # u is offset, v isn't since it's the top half

            x = uv[:, 0, :]
            y = uv[:, 1, :]
            x += xymax  # offset, since X grid starts at -xymax
            # convert to index array -- this gives the number of the uv-bin on the grid
            index = y * npixx + x
            # if we're in per-band weighting mode, then adjust the index to refer to each band's grid
            if band_mapping is not None:
                bandmap = band_mapping[iMS]
                # uv has shape nvis,nfreq; bandmap has shape nfreq
                index += bandmap[np.newaxis,:]*npix
            # zero weight refers to zero cell (otherwise it may end up outside the grid, since grid is
            # only big enough to accommodate the *unflagged* uv-points)
            index[weights==0] = 0

            weights_index[iMS] = weights_or_path, index
            del uv
            print>> log, "Accumulating weights (%d/%d)" % (iMS + 1, len(uvw_weights_flags_freqs))
            # accumulate onto grid
            # print>>log,weights,index
            _pyGridderSmearPols.pyAccumulateWeightsOntoGrid(grid, weights.ravel(), index.ravel())

        if Weighting == "uniform":
            #            print>>log,"adjusting grid to uniform weight"
            #           grid[grid!=0] = 1/grid[grid!=0]
            print>> log, ("applying uniform weighting (super=%.2f)" % Super)
            for weights_or_path, index in weights_index:
                if index is not None:
                    weights = NpShared.GiveArray(weights_or_path) if type(weights_or_path) is str else weights_or_path
                    weights /= grid[index]

        elif Weighting == "briggs" or Weighting == "robust":
            numeratorSqrt = 5.0 * 10 ** (-Robust)
            for band in range(nbands):
                print>> log, ("applying Briggs weighting (robust=%.2f, super=%.2f, band %d)" % (Robust, Super, band))
                grid1 = grid0[band,:]
                avgW = (grid1 ** 2).sum() / grid1.sum()
                sSq = numeratorSqrt ** 2 / avgW
                grid1[...] = 1 / (1 + grid1 * sSq)
            for weights_or_path, index in weights_index:
                if index is not None:
                    weights = NpShared.GiveArray(weights_or_path) if type(weights_or_path) is str else weights_or_path
                    weights *= grid[index]

        else:
            raise ValueError("unknown weighting \"%s\"" % Weighting)

        print>> log, "weights computed"


    def CalcWeightsOld(self,uvw,VisWeights,flags,freqs,Robust=0,Weighting="Briggs",Super=1):


        #u,v,_=uvw.T/*

        #Robust=-2
        nch,npol,npixIm,_=self.ImShape
        FOV=self.CellSizeRad*npixIm#/2

        #cell=1.5*4./(FOV)
        cell=1./(Super*FOV)
        #cell=4./(FOV)

        #wave=6.

        u=uvw[:,0].copy()
        v=uvw[:,1].copy()

        d=np.sqrt(u**2+v**2)
        VisWeights[d==0]=0
#        Lmean=3e8/np.mean(freqs)
        Lmin=3e8/np.max(freqs)

        uvmax=np.max(d)/Lmin
        #(1./self.CellSizeRad)#/2#np.max(d)
        npix=2*(int(uvmax/cell)+1)
        if (npix%2)==0:
            npix+=1

        #npix=npixIm
        xc,yc=npix/2,npix/2


        VisWeights=np.float64(VisWeights)
        #VisWeights.fill(1.)
        print>>log,"image grid cell is %g"%cell
        
        if Weighting=="Briggs":
            print>>log, "Weighting in Briggs mode (robust=%.1f, super=%.1f)"%(Robust,Super)
            print>>log, "Calculating imaging weights on an [%i,%i] grid"%(npix,npix)
            print>>log, ""
            Mode=0
        elif Weighting=="Uniform":
            print>>log, "Weighting in Uniform mode (super=%.1f)"%(Super)
            print>>log, "Calculating imaging weights on an [%i,%i] grid"%(npix,npix)
            Mode=1
        elif Weighting=="Natural":
            print>>log, "Weighting in Natural mode"
            return VisWeights
        else:
            stop

        grid=np.zeros((npix,npix),dtype=np.float64)


        flags=np.float32(flags)
        WW=np.mean(1.-flags,axis=2)
        VisWeights*=WW
        
        F=np.zeros(VisWeights.shape,np.int32)
        #print "u=",u
        #print "v=",v
        w=_pyGridder.pyGridderPoints(grid,
                                     F,
                                     u,
                                     v,
                                     VisWeights,
                                     float(Robust),
                                     Mode,
                                     np.float64(freqs.flatten()),
                                     np.array([cell,cell],np.float64))
        print>>log,"weights computed"


        # C=299792458.
        # uf=u.reshape((u.size,1))*freqs.reshape((1,freqs.size))/C
        # vf=v.reshape((v.size,1))*freqs.reshape((1,freqs.size))/C

        # x,y=np.int32(np.round(uf/cell))+xc,np.int32(np.round(vf/cell))+yc
        # x,y=(uf/cell)+xc,(vf/cell)+yc
        # condx=((x>0)&(x<npix))
        # condy=((y>0)&(y<npix))
        # ind=np.where((condx & condy))[0]
        # X=x#[ind]
        # Y=y#[ind]
        
        # w[w==0]=1e-10
        
        # import pylab
        # pylab.clf()
        # #pylab.scatter(uf.flatten(),vf.flatten(),c=w.flatten(),lw=0,alpha=0.3,vmin=0,vmax=1)#,w[ind,0])
        # grid[grid==0]=1e-10
        # pylab.imshow(np.log10(grid),interpolation="nearest")
        # incr=1
        # pylab.scatter(X.ravel()[::incr],Y.ravel()[::incr],c=np.log10(w.ravel())[::incr],lw=0)#,alpha=0.3)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop
        
        return w
