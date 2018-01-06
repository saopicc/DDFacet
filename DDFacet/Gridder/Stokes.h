// Form Stokes products before gridding
// See Revisiting the RIME. I. A full-sky Jones formalism
// O.M. Smirnov, 2011
// Calling conversions through casacore is too slow, so we unroll
// the most common cases here assuming the measurement set correlations
// are defined as [5,6,7,8] (RR,RL,LR,LL) or [5,8] (RR,LL) or [9,10,11,12]
// (XX,XY,YX,YY) or [9,12] (XX,YY). See
// http://www.astron.nl/casacore/trunk/casacore/doc/html/classcasa_1_1Stokes.html
// The rest remains unsupported.

using StokesGridType = void (*) (const dcMat &Vis, dcMat &stokes_vis);

#define I_FROM_XXXYYXYY (Vis[0]+Vis[3])*.5
#define Q_FROM_XXXYYXYY (Vis[0]-Vis[3])*.5
#define U_FROM_XXXYYXYY (Vis[1]+Vis[2])*.5
#define V_FROM_XXXYYXYY -(Vis[1]-Vis[2])*.5i

#define I_FROM_XXYY (Vis[0]+Vis[1])*.5
#define Q_FROM_XXYY (Vis[0]-Vis[1])*.5

#define U_FROM_XYYX (Vis[0]+Vis[1])*.5
#define V_FROM_XYYX -(Vis[0]-Vis[1])*.5i

#define I_FROM_RRLL (Vis[0]+Vis[1])*.5
#define V_FROM_RRLL (Vis[0]-Vis[1])*.5

#define Q_FROM_RLLR (Vis[0]+Vis[1])*.5
#define U_FROM_RLLR -(Vis[0]-Vis[1]).5i

#define I_FROM_RRRLLRLL (Vis[0]+Vis[3])*.5
#define Q_FROM_RRRLLRLL (Vis[1]+Vis[2])*.5
#define U_FROM_RRRLLRLL -(Vis[1]-Vis[2])*.5i
#define V_FROM_RRRLLRLL (Vis[0]-Vis[3])*.5

#define BUILD1(C0, SRC)\
void C0##_from_##SRC(const dcMat &Vis, dcMat &stokes_vis)\
  {\
  stokes_vis[0] = C0##_FROM_##SRC;\
  }
#define BUILD2(C0,C1,SRC)\
void C0##C1##_from_##SRC(const dcMat &Vis, dcMat &stokes_vis)\
  {\
  stokes_vis[0] = C0##_FROM_##SRC;\
  stokes_vis[1] = C1##_FROM_##SRC;\
  }
#define BUILD3(C0,C1,C2,SRC)\
void C0##C1##C2##_from_##SRC(const dcMat &Vis, dcMat &stokes_vis)\
  {\
  stokes_vis[0] = C0##_FROM_##SRC;\
  stokes_vis[1] = C1##_FROM_##SRC;\
  stokes_vis[2] = C2##_FROM_##SRC;\
  }
#define BUILD4(SRC)\
void IQUV_from_##SRC(const dcMat &Vis, dcMat &stokes_vis)\
  {\
  stokes_vis[0] = I_FROM_##SRC;\
  stokes_vis[1] = Q_FROM_##SRC;\
  stokes_vis[2] = U_FROM_##SRC;\
  stokes_vis[3] = V_FROM_##SRC;\
  }

BUILD1(I,RRLL)
BUILD1(V,RRLL)
BUILD2(I,V,RRLL)

//BUILD1(Q,RLLR)
//BUILD1(U,RLLR)
//BUILD2(Q,U,RLLR)

BUILD1(I,XXYY)
BUILD1(Q,XXYY)
BUILD2(I,Q,XXYY)

//BUILD1(U, XYYX)
//BUILD1(V, XYYX)
//BUILD2(U,V,XYYX)

BUILD1(I,XXXYYXYY)
BUILD1(Q,XXXYYXYY)
BUILD1(U,XXXYYXYY)
BUILD1(V,XXXYYXYY)
BUILD2(I,Q,XXXYYXYY)
BUILD2(I,U,XXXYYXYY)
BUILD2(I,V,XXXYYXYY)
BUILD2(Q,U,XXXYYXYY)
BUILD2(Q,V,XXXYYXYY)
BUILD2(U,V,XXXYYXYY)
BUILD3(I,Q,U,XXXYYXYY)
BUILD3(I,Q,V,XXXYYXYY)
BUILD3(I,U,V,XXXYYXYY)
BUILD3(Q,U,V,XXXYYXYY)
BUILD4(XXXYYXYY)

BUILD1(I,RRRLLRLL)
BUILD1(Q,RRRLLRLL)
BUILD1(U,RRRLLRLL)
BUILD1(V,RRRLLRLL)
BUILD2(I,Q,RRRLLRLL)
BUILD2(I,U,RRRLLRLL)
BUILD2(I,V,RRRLLRLL)
BUILD2(Q,U,RRRLLRLL)
BUILD2(Q,V,RRRLLRLL)
BUILD2(U,V,RRRLLRLL)
BUILD3(I,Q,U,RRRLLRLL)
BUILD3(I,Q,V,RRRLLRLL)
BUILD3(I,U,V,RRRLLRLL)
BUILD3(Q,U,V,RRRLLRLL)
BUILD4(RRRLLRLL)

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
  corr_vis[0] = corr_vis[1] = stokes_vis[0];\
  corr_vis[2] = corr_vis[3] = 0.;
#define XXXYYXYY_FROM_I\
  corr_vis[0] = corr_vis[3] = stokes_vis[0];\
  corr_vis[1] = corr_vis[2] = 0.;
#define RRLL_FROM_I\
  corr_vis[0] = corr_vis[1] = stokes_vis[0];\
  corr_vis[2] = corr_vis[3] = 0.;
#define RRRLLRLL_FROM_I\
  corr_vis[0] = corr_vis[3] = stokes_vis[0];\
  corr_vis[1] = corr_vis[2] = 0.;

typedef void(*StokesDegridType)(const dcMat &stokes_vis, dcMat &corr_vis);

#define PUT1(NAME, COMP)\
void NAME (const dcMat &stokes_vis, dcMat &corr_vis)\
  {\
  COMP\
  }

PUT1(gmode_corr_XXYY_from_I, XXYY_FROM_I)
PUT1(gmode_corr_XXXYYXYY_from_I, XXXYYXYY_FROM_I)
PUT1(gmode_corr_RRLL_from_I, RRLL_FROM_I)
PUT1(gmode_corr_RRRLLRLL_from_I, RRRLLRLL_FROM_I)
