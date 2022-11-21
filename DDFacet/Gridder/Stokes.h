// Form Stokes products before gridding
// See Revisiting the RIME. I. A full-sky Jones formalism
// O.M. Smirnov, 2011
// Calling conversions through casacore is too slow, so we unroll
// the most common cases here assuming the measurement set correlations
// are defined as [5,6,7,8] (RR,RL,LR,LL) or [5,8] (RR,LL) or [9,10,11,12]
// (XX,XY,YX,YY) or [9,12] (XX,YY). See
// http://www.astron.nl/casacore/trunk/casacore/doc/html/classcasa_1_1Stokes.html
// The rest remains unsupported.

#ifndef GRIDDER_STOKES_H
#define GRIDDER_STOKES_H

#include <cmath>
namespace DDF {
  using namespace std;

  #define I_FROM_XXXYYXYY (Vis[0]+Vis[3])*.5
  #define Q_FROM_XXXYYXYY (Vis[0]-Vis[3])*.5
  #define U_FROM_XXXYYXYY (Vis[1]+Vis[2])*.5
  #define V_FROM_XXXYYXYY -(Vis[1]-Vis[2])*.5i

  #define I_FROM_XXYY (Vis[0]+Vis[3])*.5
  #define Q_FROM_XXYY (Vis[0]-Vis[3])*.5

  #define I_FROM_RRLL (Vis[0]+Vis[3])*.5
  #define V_FROM_RRLL (Vis[0]-Vis[3])*.5

  #define I_FROM_RRRLLRLL (Vis[0]+Vis[3])*.5
  #define Q_FROM_RRRLLRLL (Vis[1]+Vis[2])*.5
  #define U_FROM_RRRLLRLL -(Vis[1]-Vis[2])*.5i
  #define V_FROM_RRRLLRLL (Vis[0]-Vis[3])*.5

  #define I_FROM_I (Vis[0])

  #define BUILD1(C0, SRC)\
  inline dcMat C0##_from_##SRC(const dcMat &Vis)\
    { return dcMat(C0##_FROM_##SRC,0,0,0); }
  #define BUILD2(C0,C1,SRC)\
  inline dcMat C0##C1##_from_##SRC(const dcMat &Vis)\
    { return dcMat(C0##_FROM_##SRC,C1##_FROM_##SRC,0,0); }
  #define BUILD3(C0,C1,C2,SRC)\
  inline dcMat C0##C1##C2##_from_##SRC(const dcMat &Vis)\
    { return dcMat(C0##_FROM_##SRC,C1##_FROM_##SRC,C2##_FROM_##SRC,0); }
  #define BUILD4(SRC)\
  inline dcMat IQUV_from_##SRC(const dcMat &Vis)\
    { return dcMat(I_FROM_##SRC,Q_FROM_##SRC,U_FROM_##SRC,V_FROM_##SRC); }
  namespace gridder {
    namespace policies {
      using StokesGridType = dcMat (*) (const dcMat &Vis);
      BUILD1(I,I)
      
      BUILD1(I,RRLL)
      BUILD2(I,V,RRLL)

      BUILD1(I,XXYY)
      BUILD2(I,Q,XXYY)

      BUILD1(I,XXXYYXYY)
      BUILD1(Q,XXXYYXYY)
      BUILD1(U,XXXYYXYY)
      BUILD1(V,XXXYYXYY)
      BUILD2(I,Q,XXXYYXYY)
      BUILD2(I,V,XXXYYXYY)
      BUILD2(Q,U,XXXYYXYY)
      BUILD3(I,Q,U,XXXYYXYY)
      BUILD4(XXXYYXYY)

      BUILD1(I,RRRLLRLL)
      BUILD2(I,Q,RRRLLRLL)
      BUILD2(I,U,RRRLLRLL)
      BUILD2(I,V,RRRLLRLL)
      BUILD2(Q,U,RRRLLRLL)
      BUILD3(I,Q,U,RRRLLRLL)
      BUILD4(RRRLLRLL)
    }
  }

  //--------------------------------------------
  // Inverse operations: correlations from stokes
  //
  // WARNING:
  // Currently only Stokes I deconvolution is supported
  // So there will always be either the V fraction missing (expected to be minimal)
  // in the case of circular feeds and Q in the case of
  // linear feeds (may not be so negligible on sensitive arrays).
  // This section must be greatly expanded when full polarization
  // cleaning is supported in the future.
  //--------------------------------------------
  #define XXYY_FROM_I\
    return dcMat(stokes_vis[0],0,0,stokes_vis[0]);
  #define I_FROM_I\
    return dcMat(stokes_vis[0],0,0,stokes_vis[0]);
  #define XXXYYXYY_FROM_I\
    return dcMat(stokes_vis[0],0,0,stokes_vis[0]);
  #define RRLL_FROM_I\
    return dcMat(stokes_vis[0],0,0,stokes_vis[0]);
  #define RRRLLRLL_FROM_I\
    return dcMat(stokes_vis[0],0,0,stokes_vis[0]);
  // newly added by FG 20210610
  #define RRRLLRLL_FROM_IQUV\
    return dcMat((stokes_vis[0] + stokes_vis[3]),\
                 (stokes_vis[1] + 1i*stokes_vis[2]),\
                 (stokes_vis[1] - 1i*stokes_vis[2]),\
                 (stokes_vis[0] - stokes_vis[3]));
  #define RRRLLRLL_FROM_IQ\
    return dcMat((stokes_vis[0] + 0.0),\
                 (stokes_vis[1]),\
                 (stokes_vis[1]),\
                 (stokes_vis[0] - 0.0));
  #define RRRLLRLL_FROM_IU\
    return dcMat((stokes_vis[0] + 0.),\
                 (0. + 1i*stokes_vis[1]),\
                 (0. - 1i*stokes_vis[1]),\
                 (stokes_vis[0] - 0.));
  #define RRRLLRLL_FROM_IV\
    return dcMat((stokes_vis[0] + stokes_vis[1]),\
                 (stokes_vis[0] * 0. + 0.),\
                 (stokes_vis[0] * 0. - 0.),\
                 (stokes_vis[0] - stokes_vis[1]));
  #define RRRLLRLL_FROM_IQU\
    return dcMat((stokes_vis[0] + 0.),\
                 (stokes_vis[1] + 1i*stokes_vis[2]),\
                 (stokes_vis[1] - 1i*stokes_vis[2]),\
                 (stokes_vis[0] - 0.));
  #define XXXYYXYY_FROM_IQUV\
    return dcMat((stokes_vis[0] + stokes_vis[1]),\
                 (stokes_vis[2] + 1i*stokes_vis[3]),\
                 (stokes_vis[2] - 1i*stokes_vis[3]),\
                 (stokes_vis[0] - stokes_vis[1]));
  // for the following cases we assume the user is not really interested
  // in doing any polarimetry, but wants to build a model for Q or V
  // in order not to bias solutions during selfcal
  // I +/- V circular
  // I +/- Q linear
  #define XXXYYXYY_FROM_IQ\
    return dcMat((stokes_vis[0] + stokes_vis[1]),\
                 (stokes_vis[0] * 0.),\
                 (stokes_vis[0] * 0.),\
                 (stokes_vis[0] - stokes_vis[1]));
  #define XXXYYXYY_FROM_IU\
    return dcMat((stokes_vis[0]),\
                 (stokes_vis[1]),\
                 (stokes_vis[1]),\
                 (stokes_vis[0]));
  #define XXXYYXYY_FROM_IV\
    return dcMat((stokes_vis[0] + 0.),\
                 (0. + stokes_vis[1] * 1i),\
                 (0. - stokes_vis[1] * 1i),\
                 (stokes_vis[0] - 0.));
  #define XXXYYXYY_FROM_V\
    return dcMat((stokes_vis[0] * 0.),\
                 (0. + stokes_vis[0] * 1i),\
                 (0. - stokes_vis[0] * 1i),\
                 (stokes_vis[0] * 0.));
  #define XXXYYXYY_FROM_IQU\
    return dcMat((stokes_vis[0] + stokes_vis[1]),\
                 (stokes_vis[2]),\
                 (stokes_vis[2]),\
                 (stokes_vis[0] - stokes_vis[1]));
  #define XXXYYXYY_FROM_QU\
    return dcMat((0. + stokes_vis[0]),\
                 (stokes_vis[1]),\
                 (stokes_vis[1]),\
                 (0. - stokes_vis[0]));
  #define XXXYYXYY_FROM_Q\
    return dcMat((0. + stokes_vis[0]),\
                 (stokes_vis[0] * 0.),\
                 (stokes_vis[0] * 0.),\
                 (0. - stokes_vis[0]));
  #define XXXYYXYY_FROM_U\
    return dcMat((stokes_vis[0] * 0.),\
                 (stokes_vis[0]),\
                 (stokes_vis[0]),\
                 (stokes_vis[0] * 0.));
  #define RRRLLRLL_FROM_IV\
    return dcMat((stokes_vis[0] + stokes_vis[1]),\
                 (stokes_vis[0] * 0.),\
                 (stokes_vis[0] * 0.),\
                 (stokes_vis[0] - stokes_vis[1]));
  // Dual correlation cases for the above comment  
  #define RRLL_FROM_IV\
    return dcMat((stokes_vis[0] + stokes_vis[1]),\
                 (stokes_vis[0] * 0.),\
                 (stokes_vis[0] * 0.),\
                 (stokes_vis[0] - stokes_vis[1]));
  
  #define XXYY_FROM_IQ\
    return dcMat((stokes_vis[0] + stokes_vis[1]),\
                 (stokes_vis[0] * 0.),\
                 (stokes_vis[0] * 0.),\
                 (stokes_vis[0] - stokes_vis[1]));

  using StokesDegridType = dcMat (*)(const dcMat &stokes_vis);

  #define PUT1(NAME, COMP)\
  inline dcMat NAME (const dcMat &stokes_vis)\
    { COMP }
  namespace degridder {
    namespace policies {
      using StokesDegridType = dcMat (*)(const dcMat &stokes_vis);
      PUT1(gmode_corr_I_from_I, I_FROM_I)
      PUT1(gmode_corr_XXYY_from_I, XXYY_FROM_I)
      PUT1(gmode_corr_XXXYYXYY_from_I, XXXYYXYY_FROM_I)
      PUT1(gmode_corr_RRLL_from_I, RRLL_FROM_I)
      PUT1(gmode_corr_RRRLLRLL_from_I, RRRLLRLL_FROM_I)
      //newly added by FG 20210610
      PUT1(gmode_corr_RRRLLRLL_from_IQUV, RRRLLRLL_FROM_IQUV)
      PUT1(gmode_corr_RRRLLRLL_from_IQ, RRRLLRLL_FROM_IQ)
      PUT1(gmode_corr_RRRLLRLL_from_IU, RRRLLRLL_FROM_IU)
      PUT1(gmode_corr_RRRLLRLL_from_IQU, RRRLLRLL_FROM_IQU)
      PUT1(gmode_corr_XXXYYXYY_from_IQUV, XXXYYXYY_FROM_IQUV)
      //further dual corr policies by BH 20220222
      PUT1(gmode_corr_RRRLLRLL_from_IV, RRRLLRLL_FROM_IV)
      PUT1(gmode_corr_XXXYYXYY_from_IQ, XXXYYXYY_FROM_IQ)
      PUT1(gmode_corr_XXXYYXYY_from_IV, XXXYYXYY_FROM_IV)      
      PUT1(gmode_corr_XXXYYXYY_from_IU, XXXYYXYY_FROM_IU)
      PUT1(gmode_corr_XXXYYXYY_from_U, XXXYYXYY_FROM_U)
      PUT1(gmode_corr_XXXYYXYY_from_Q, XXXYYXYY_FROM_Q)
      PUT1(gmode_corr_XXXYYXYY_from_QU, XXXYYXYY_FROM_QU)
      PUT1(gmode_corr_XXXYYXYY_from_IQU, XXXYYXYY_FROM_IQU)
      PUT1(gmode_corr_XXXYYXYY_from_V, XXXYYXYY_FROM_V)
      PUT1(gmode_corr_RRLL_from_IV, RRLL_FROM_IV)
      PUT1(gmode_corr_XXYY_from_IQ, XXYY_FROM_IQ)
    }
  }
}
#endif /*GRIDDER_STOKES_H*/
