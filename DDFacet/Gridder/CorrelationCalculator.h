#pragma once
#include "common.h"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#ifndef GRIDDER_CORRCALC_H
#define GRIDDER_CORRCALC_H

namespace DDF {
  namespace py=pybind11;
  using namespace std;
  class CorrectionCalculator
    {
    private:
      bool ChanEquidistant;
      /* these are used for equidistant channels: one holds the phase term in the
         current channel, the other one holds the delta-phase across channels */
      dcmplx CurrentCorrTerm, dCorrTerm;
      /* and this indicates for which channel the CurrentCorrTerm is currently computed */
      int CurrentCorrChan;

    public:
      CorrectionCalculator(const py::list& LOptimisation)
        {
        ChanEquidistant= LOptimisation[1].cast<bool>();
        if (!ChanEquidistant) return;
        CurrentCorrChan = -1;
        }

      void update()
        {
        CurrentCorrChan = -1;
        }

      dcmplx getCorr(const double *Pfreqs, size_t visChan, double angle)
        {
        if (!ChanEquidistant)
          return polar(1.,Pfreqs[visChan]*angle);

        /* init correlation term for first channel that it's not initialized in */
        if (CurrentCorrChan == -1)
          {
          CurrentCorrTerm = polar(1.,Pfreqs[visChan]*angle);
          dCorrTerm       = polar(1.,(Pfreqs[1]-Pfreqs[0])*angle);
          CurrentCorrChan = int(visChan);
          }
        /* else, wind the correlation term forward by as many channels as necessary */
        /* this modification allows us to support blocks that skip across channels */
        else if( CurrentCorrChan > int(visChan) )
          {
          cerr<<"Reverse channel ordering. ("<<CurrentCorrChan<<", "<<visChan<<") This must be a bug in the BDA mapping.\n";
          abort();
          }
        else
          {
          while (size_t(CurrentCorrChan)<visChan)
            {
            CurrentCorrTerm *= dCorrTerm;
            CurrentCorrChan++;
            }
          }
        return CurrentCorrTerm;
        }
    };
}

#endif /*GRIDDER_CORRCALC_H*/
