#include "CorrelationCalculator.h"

namespace DDF {
  CorrectionCalculator::CorrectionCalculator(const py::list& LOptimisation, 
					     const bool *sparsificationFlag_,
					     const size_t NTotBlocks, 
					     const int *NRowBlocks)
    {
    ChanEquidistant= LOptimisation[1].cast<bool>();
    if (!ChanEquidistant) return;

    sparsificationFlag = sparsificationFlag_;
    NMaxRow=0;
    for (size_t i=0; i<NTotBlocks; ++i)
      if (!sparsificationFlag || sparsificationFlag[i])
	NMaxRow = max(NMaxRow, size_t(NRowBlocks[i]-2));
    CurrentCorrTerm.resize(NMaxRow);
    dCorrTerm.resize(NMaxRow);
    CurrentCorrChan.resize(NMaxRow,-1);
    CurrentCorrRow0 = -1;
    }

  void CorrectionCalculator::update(int Row0, int NRowThisBlock)
    {
    /* when moving to a new block of rows, init this to -1 so the code below knows to initialize*/
    /* CurrentCorrTerm when the first channel of each row comes in*/
    if ((!ChanEquidistant) || (Row0==CurrentCorrRow0)) return;
    fill_n(CurrentCorrChan.begin(), NRowThisBlock, -1);
    CurrentCorrRow0 = Row0;
    }

  dcmplx CorrectionCalculator::getCorr(int inx, const double *Pfreqs, size_t visChan, double angle)
    {
    if (!ChanEquidistant)
      return polar(1.,Pfreqs[visChan]*angle);

    /* init correlation term for first channel that it's not initialized in */
    if (CurrentCorrChan[inx]==-1)
      {
      CurrentCorrTerm[inx] = polar(1.,Pfreqs[visChan]*angle);
      dCorrTerm[inx]       = polar(1.,(Pfreqs[1]-Pfreqs[0])*angle);
      CurrentCorrChan[inx] = int(visChan);
      }
    /* else, wind the correlation term forward by as many channels as necessary */
    /* this modification allows us to support blocks that skip across channels */
    else
      while (size_t(CurrentCorrChan[inx])<visChan)
	{
	CurrentCorrTerm[inx] *= dCorrTerm[inx];
	CurrentCorrChan[inx]++;
	}
    return CurrentCorrTerm[inx];
    }
}