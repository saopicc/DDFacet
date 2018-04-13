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
      const bool *sparsificationFlag;
      size_t NMaxRow;
      /* these are used for equidistant channels: one holds the phase term in the
	current channel, the other one holds the delta-phase across channels */
      vector<dcmplx> CurrentCorrTerm, dCorrTerm;
      /* and this indicates for which channel the CurrentCorrTerm is currently computed */
      vector<int> CurrentCorrChan;
      int CurrentCorrRow0;

    public:
      CorrectionCalculator(const py::list& LOptimisation, 
			   const bool *sparsificationFlag_,
			   const size_t NTotBlocks, 
			   const int *NRowBlocks);

      void update(int Row0, int NRowThisBlock);
      dcmplx getCorr(int inx, const double *Pfreqs, size_t visChan, double angle);
    };
}

#endif GRIDDER_CORRCALC_H