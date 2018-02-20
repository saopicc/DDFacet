// Form Stokes products before gridding 
// See Revisiting the RIME. I. A full-sky Jones formalism 
// O.M. Smirnov, 2011 
// Calling conversions through casacore is too slow, so we unroll 
// the most common cases here assuming the measurement set correlations 
// are defined as [5,6,7,8] (RR,RL,LR,LL) or [5,8] (RR,LL) or [9,10,11,12]  
// (XX,XY,YX,YY) or [9,12] (XX,YY). See  
// http://www.astron.nl/casacore/trunk/casacore/doc/html/classcasa_1_1Stokes.html 
// The rest remains unsupported. 
#include "complex.h"
#define GMODE_STOKES_I_FROM_XXXYYXYY  \
	float _Complex stokes_vis[] = {0+0*_Complex_I};  \
	stokes_vis[0]=(Vis[0]+Vis[3])/2.;  \
	int nVisPol = 1;  \
	int PolMap[] = {0}; 
#define GMODE_STOKES_I_FROM_XXYY  \
	float _Complex stokes_vis[] = {0+0*_Complex_I};  \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_IQ_FROM_XXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	stokes_vis[1]=(Vis[0]-Vis[1])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_QI_FROM_XXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	stokes_vis[1]=(Vis[0]-Vis[1])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_I_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[3])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_I_FROM_RRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_IV_FROM_RRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[1])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[1])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_VI_FROM_RRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[1])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[1])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_Q_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_U_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_V_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_V_FROM_XYYX \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(-(Vis[0]-Vis[1])*_Complex_I)/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0};
#define GMODE_STOKES_U_FROM_XYYX \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_Q_FROM_RLLR \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_QU_FROM_RLLR \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	stokes_vis[1]=(-(Vis[0]-Vis[1])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_UQ_FROM_RLLR \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	stokes_vis[1]=(-(Vis[0]-Vis[1])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_UV_FROM_XYYX \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	stokes_vis[1]=(-(Vis[0]-Vis[1])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1};
#define GMODE_STOKES_VU_FROM_XYYX \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0]+Vis[1])/2.; \
	stokes_vis[1]=(-(Vis[0]-Vis[1])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_U_FROM_RLLR \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(-(Vis[0]-Vis[1])*_Complex_I)/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_V_FROM_RRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[1])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_V_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_Q_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_Q_FROM_XXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[1])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_U_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 1; \
	int PolMap[] = {0}; 
#define GMODE_STOKES_IQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_QI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_IU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_UI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_IV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_VI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_UQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[1]+Vis[2])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_QU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[1]+Vis[2])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_QV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_VQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_UV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[1]+Vis[2])/2.; \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1};
#define GMODE_STOKES_VU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[1]+Vis[2])/2.; \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_IQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_QI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_IU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_UI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_IV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_VI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_UQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_QU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_QV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[1]+Vis[2])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1}; 
#define GMODE_STOKES_VQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[1]+Vis[2])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_UV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {0,1};
#define GMODE_STOKES_VU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	int nVisPol = 2; \
	int PolMap[] = {1,0}; 
#define GMODE_STOKES_IQU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 3; \
	int PolMap[] = {0,1,2}; 
#define GMODE_STOKES_IUQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 3; \
	int PolMap[] = {0,2,1}; 
#define GMODE_STOKES_UIQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 3; \
	int PolMap[] = {1,2,0};
#define GMODE_STOKES_UQI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 3; \
	int PolMap[] = {2,1,0};
#define GMODE_STOKES_QUI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 3; \
	int PolMap[] = {2,0,1};
#define GMODE_STOKES_QIU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; \
	int nVisPol = 3; \
	int PolMap[] = {1,0,2};
#define GMODE_STOKES_IQV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.;\
	int nVisPol = 3; \
	int PolMap[] = {0,1,2};
#define GMODE_STOKES_IVQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.;\
	int nVisPol = 3; \
	int PolMap[] = {0,2,1};
#define GMODE_STOKES_VIQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.;\
	int nVisPol = 3; \
	int PolMap[] = {1,2,0};
#define GMODE_STOKES_VQI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.;\
	int nVisPol = 3; \
	int PolMap[] = {2,1,0};
#define GMODE_STOKES_QVI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.;\
	stokes_vis[1]=(Vis[0] - Vis[3])/2.;\
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.;\
	int nVisPol = 3; \
	int PolMap[] = {2,0,1};
#define GMODE_STOKES_QIV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,0,2};
#define GMODE_STOKES_IUV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1] + Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,1,2};
#define GMODE_STOKES_IVU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,2,1};
#define GMODE_STOKES_VIU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,2,0};
#define GMODE_STOKES_VUI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,1,0};
#define GMODE_STOKES_UVI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,0,1};
#define GMODE_STOKES_UIV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,0,2};
#define GMODE_STOKES_QUV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,1,2};
#define GMODE_STOKES_QVU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,2,1};
#define GMODE_STOKES_VQU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,2,0};
#define GMODE_STOKES_VUQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,1,0};
#define GMODE_STOKES_UVQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,0,1};
#define GMODE_STOKES_UQV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,0,2};
#define GMODE_STOKES_IQUV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,1,2,3};
#define GMODE_STOKES_IQVU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,1,2,3};
#define GMODE_STOKES_IUQV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,2,1,3};
#define GMODE_STOKES_IUVQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,3,1,2};
#define GMODE_STOKES_IVQU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,2,3,1};
#define GMODE_STOKES_IVUQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,3,2,1};
#define GMODE_STOKES_QIUV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,0,2,3};
#define GMODE_STOKES_QIVU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,0,3,2};
#define GMODE_STOKES_VIUQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,3,2,0};
#define GMODE_STOKES_VIQU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,2,3,0};
#define GMODE_STOKES_UIVQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,3,0,2};
#define GMODE_STOKES_UIQV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,2,0,3};
#define GMODE_STOKES_QUIV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,0,1,3};
#define GMODE_STOKES_UQIV_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,1,0,3};
#define GMODE_STOKES_UVIQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,3,0,1};
#define GMODE_STOKES_VUIQ_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,3,1,0};
#define GMODE_STOKES_VQIU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,1,3,0};
#define GMODE_STOKES_QVIU_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,0,3,1};
#define GMODE_STOKES_QUVI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,1,0,2};
#define GMODE_STOKES_UQVI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,1,0,2};
#define GMODE_STOKES_UVQI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,2,0,1};
#define GMODE_STOKES_VUQI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,2,1,0};
#define GMODE_STOKES_VQUI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,1,2,0};
#define GMODE_STOKES_QVUI_FROM_XXXYYXYY \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[0] - Vis[3])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[1]+Vis[2])/2.; /*U*/ \
	stokes_vis[3]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,0,1,2};
#define GMODE_STOKES_IQU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,1,2};
#define GMODE_STOKES_IUQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,2,1};
#define GMODE_STOKES_UIQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,2,0};
#define GMODE_STOKES_UQI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,1,0};
#define GMODE_STOKES_QUI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,0,1};
#define GMODE_STOKES_QIU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,0,2};
#define GMODE_STOKES_IQV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,1,2};
#define GMODE_STOKES_IVQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,2,1};
#define GMODE_STOKES_VIQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,2,0};
#define GMODE_STOKES_VQI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,1,0};
#define GMODE_STOKES_QVI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,0,1};
#define GMODE_STOKES_QIV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,0,2};
#define GMODE_STOKES_IUV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,1,2};
#define GMODE_STOKES_IVU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,2,1};
#define GMODE_STOKES_VIU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,2,0};
#define GMODE_STOKES_VUI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,1,0};
#define GMODE_STOKES_UVI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,0,1};
#define GMODE_STOKES_UIV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,0,2};
#define GMODE_STOKES_QUV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,1,2};
#define GMODE_STOKES_QVU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {0,2,1};
#define GMODE_STOKES_VQU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,2,0};
#define GMODE_STOKES_VUQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,1,0};
#define GMODE_STOKES_UVQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {2,0,1};
#define GMODE_STOKES_UQV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[1]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[2]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 3; \
	int PolMap[] = {1,0,2};
#define GMODE_STOKES_IQUV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,1,2,3};
#define GMODE_STOKES_IQVU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,1,2,3};
#define GMODE_STOKES_IUQV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,2,1,3};
#define GMODE_STOKES_IUVQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,3,1,2};
#define GMODE_STOKES_IVQU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,2,3,1};
#define GMODE_STOKES_IVUQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {0,3,2,1};
#define GMODE_STOKES_QIUV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,0,2,3};
#define GMODE_STOKES_QIVU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,0,3,2};
#define GMODE_STOKES_VIUQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,3,2,0};
#define GMODE_STOKES_VIQU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,2,3,0};
#define GMODE_STOKES_UIVQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,3,0,2};
#define GMODE_STOKES_UIQV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {1,2,0,3};
#define GMODE_STOKES_QUIV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,0,1,3};
#define GMODE_STOKES_UQIV_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,1,0,3};
#define GMODE_STOKES_UVIQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,3,0,1};
#define GMODE_STOKES_VUIQ_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,3,1,0};
#define GMODE_STOKES_VQIU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,1,3,0};
#define GMODE_STOKES_QVIU_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {2,0,3,1};
#define GMODE_STOKES_QUVI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,1,0,2};
#define GMODE_STOKES_UQVI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,1,0,2};
#define GMODE_STOKES_UVQI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,2,0,1};
#define GMODE_STOKES_VUQI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,2,1,0};
#define GMODE_STOKES_VQUI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,1,2,0};
#define GMODE_STOKES_QVUI_FROM_RRRLLRLL \
	float _Complex stokes_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	stokes_vis[0]=(Vis[0] + Vis[3])/2.; /*I*/ \
	stokes_vis[1]=(Vis[1]+Vis[2])/2.; /*Q*/ \
	stokes_vis[2]=(-(Vis[1]-Vis[2])*_Complex_I)/2.; /*U*/ \
	stokes_vis[3]=(Vis[0] - Vis[3])/2.; /*V*/ \
	int nVisPol = 4; \
	int PolMap[] = {3,0,1,2};
//--------------------------------------------
// Inverse operations: correlations from stokes
//
// WARNING:
// Currently only Stokes I deconvolution is supported
// So there will always be either the V fraction missing (expected to be minimal)
// in the case of circular feeds and Q in the case of
// linear feeds (may not be so negligiable on sensitive arrays). 
// This section must be greatly expanded when full polarization
// cleaning is supported in the future.
//--------------------------------------------
#define GMODE_CORR_RRLL_FROM_I \
	float _Complex corr_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	/* I missing V*/ \
	corr_vis[0] = stokes_vis[0]; \
	corr_vis[1] = stokes_vis[0]; \
	int nVisCorr = 2; 
	
#define GMODE_CORR_RRRLLRLL_FROM_I \
	float _Complex corr_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	/* I missing V*/ \
	corr_vis[0] = stokes_vis[0]; \
	corr_vis[3] = stokes_vis[0]; \
	int nVisCorr = 4; 
	
#define GMODE_CORR_XXYY_FROM_I \
	float _Complex corr_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	/* I missing Q*/ \
	corr_vis[0] = stokes_vis[0]; \
	corr_vis[1] = stokes_vis[0]; \
	int nVisCorr = 2;

#define GMODE_CORR_XXXYYXYY_FROM_I \
	float _Complex corr_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
	/* I missing Q*/ \
	corr_vis[0] = stokes_vis[0]; \
	corr_vis[3] = stokes_vis[0]; \
	int nVisCorr = 4;
