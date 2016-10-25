#include <Python.h>
#include <math.h>
#include <time.h>
#include "arrayobject.h"
#include "complex.h"
#include <omp.h>
#include "Matrix.c"

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

void ScaleJones(float complex* J0, float AlphaScaleJones){
  float complex z0;
  int ThisPol;
  int nPol=4;
  if(FullScalarMode){nPol=1;}

  for(ThisPol =0; ThisPol<nPol;ThisPol++){
    if(cabs(J0[ThisPol])!=0.){
      z0=J0[ThisPol]/cabs(J0[ThisPol]);
      J0[ThisPol]=(1.-AlphaScaleJones)*z0+AlphaScaleJones*J0[ThisPol];
      //J0[ThisPol]=z0+AlphaScaleJones*(J0[ThisPol]-z0);
    }
  }
}





void NormJones(float complex* J0, int ApplyAmp, int ApplyPhase, int DoScaleJones, double *uvwPtr, float WaveLengthMean, float CalibError){
  int ThisPol;
  int nPol=4;
  if(FullScalarMode){nPol=1;}
  if(ApplyAmp==0){
    for(ThisPol =0; ThisPol<nPol;ThisPol++){
      if(cabs(J0[ThisPol])!=0.){
	J0[ThisPol]/=cabs(J0[ThisPol]);
      }
    }
  }
	
  if(ApplyPhase==0){
    for(ThisPol =0; ThisPol<nPol;ThisPol++){
      J0[ThisPol]=cabs(J0[ThisPol]);
    }
  }
	
  if(DoScaleJones==1){
    float U2=uvwPtr[0]*uvwPtr[0];
    float V2=uvwPtr[1]*uvwPtr[1];
    float R2=(U2+V2)/(WaveLengthMean*WaveLengthMean);
    float CalibError2=CalibError*CalibError;
    float AlphaScaleJones=exp(-2.*PI*CalibError2*R2);
    ScaleJones(J0,AlphaScaleJones);
  }
}

void PrintArray(float complex *A){
  printf("===================\n");
  printf("[[(%f %f 1j)   ",creal(A[0]),cimag(A[0]));
  printf("(%f %f 1j)] \n",creal(A[1]),cimag(A[1]));
  printf(" [(%f %f 1j)   ",creal(A[2]),cimag(A[2]));
  printf("(%f %f 1j)]] \n",creal(A[3]),cimag(A[3]));
}


void GiveJones(float complex *ptrJonesMatrices, int *JonesDims, float *ptrCoefs, int i_t, int i_ant0, int i_dir, int iChJones, int Mode, float complex *Jout){
  size_t nd_Jones,na_Jones,nch_Jones;
  nd_Jones=JonesDims[1];
  na_Jones=JonesDims[2];
  nch_Jones=JonesDims[3];
  
  int nPol=4;
  //int iChJones=0;
  if(FullScalarMode){nPol=1;}
  int ipol;
  
  if(Mode==0){
    size_t offJ0=i_t*nd_Jones*na_Jones*nch_Jones*4
      +i_dir*na_Jones*nch_Jones*4
      +i_ant0*nch_Jones*4
      +iChJones*4;
    
    //printf("%i %i %i %i -> %f %f\n",i_t,i_dir,i_ant0,iChJones,creal(*(ptrJonesMatrices+offJ0)),cimag(*(ptrJonesMatrices+offJ0)));
    //PrintArray(ptrJonesMatrices+offJ0);
    
    for(ipol=0; ipol<nPol; ipol++){
      Jout[ipol]=*(ptrJonesMatrices+offJ0+ipol);
    }
  }


  float Jabs[4]={0};
  float A=0;
  
  if(Mode==1){
    int ndone=0;
    for(ipol=0; ipol<nPol; ipol++){
      Jout[ipol]=0;
    }

    int idir;
    for(idir=0; idir<nd_Jones; idir++){
      if(ptrCoefs[idir]==0){continue;}
      size_t offJ0=i_t*nd_Jones*na_Jones*nch_Jones*4
	+i_dir*na_Jones*nch_Jones*4
	+i_ant0*nch_Jones*4;
	+iChJones*4;

      float coef;
      float complex val;
      for(ipol=0; ipol<nPol; ipol++){
	A=cabs(*(ptrJonesMatrices+offJ0+ipol));
	//Jout[ipol]+=((float)(ptrCoefs[idir])/(float)(A))*(*(ptrJonesMatrices+offJ0+ipol));
	Jout[ipol]+=(ptrCoefs[idir]/A)*(*(ptrJonesMatrices+offJ0+ipol));
	Jabs[ipol]+=ptrCoefs[idir]*A;
	//printf("[%i, %i] coef=%f val=(%f,%f) J=(%f,%f) A=(%f,%f) \n",ipol,ndone,coef,creal(val),cimag(val),creal(Jout[ipol]),cimag(Jout[ipol]),creal(Jabs[ipol]),cimag(Jabs[ipol]));
      }
      ndone+=1;

      //printf("w=(%f) A=%f \n",ptrCoefs[idir],A);
    }//end for idir

    for(ipol=0; ipol<nPol; ipol++){
      Jout[ipol]*=Jabs[ipol];
      //printf("[%i, %i] J=(%f,%f) A=(%f,%f) \n",ipol,ndone,creal(Jout[ipol]),cimag(Jout[ipol]),creal(Jabs[ipol]),cimag(Jabs[ipol]));
    }
    

    
  }//endif


}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

int DoApplyJones=0;
PyArrayObject *npJonesMatrices, *npTimeMappingJonesMatrices, *npA0, *npA1, *npJonesIDIR, *npCoefsInterp,*npModeInterpolation;
float complex* ptrJonesMatrices;
int *ptrTimeMappingJonesMatrices,*ptrA0,*ptrA1,*ptrJonesIDIR;
float *ptrCoefsInterp;
int i_dir_kMS;
int nd_Jones,na_Jones,nch_Jones,nt_Jones;

PyArrayObject *npAlphaReg_killMS;
float* ptrAlphaReg_killMS;

PyArrayObject *npJonesMatrices_Beam, *npTimeMappingJonesMatrices_Beam;
PyArrayObject *npVisToJonesChanMapping_killMS,*npVisToJonesChanMapping_Beam;
int *ptrVisToJonesChanMapping_killMS,*ptrVisToJonesChanMapping_Beam;
float complex* ptrJonesMatrices_Beam;
int *ptrTimeMappingJonesMatrices_Beam;
int nd_Jones_Beam,na_Jones_Beam,nch_Jones_Beam,nt_Jones_Beam;
int nd_AlphaReg,na_AlphaReg;
int JonesDims_Beam[4];
int ApplyJones_Beam=0;
int i_dir_Beam;
PyArrayObject *npJonesIDIR_Beam;
int *ptrJonesIDIR_Beam;
int ApplyJones_killMS=0; 

//    printf("len %i",LengthJonesList);
int JonesDims[4];
int ModeInterpolation=1;
int *ptrModeInterpolation;
int ApplyAmp,ApplyPhase,DoScaleJones;
float CalibError,CalibError2,ReWeightSNR;
double *ptrSumJones;
double *ptrSumJonesChan;

float complex *J0;
float complex *J1;
float complex *J0kMS;
float complex *J1kMS;
float complex *J0Beam;
float complex *J1Beam;


float complex *J0inv;
float complex *J0H;
float complex *J0Conj;
float complex *J1Conj;
float complex *J1H;
float complex *J1T;
float complex *J1Hinv;
float complex *JJ;
float complex *J0kMS_tp1;
float complex *J1kMS_tp1;
float complex *J0kMS_tm1;
float complex *J1kMS_tm1;
float complex *IMatrix;


void initJonesMatrices(){
  J0=calloc(1,(4)*sizeof(float complex));
  J1=calloc(1,(4)*sizeof(float complex));
  J0kMS=calloc(1,(4)*sizeof(float complex));
  J1kMS=calloc(1,(4)*sizeof(float complex));
  J0Beam=calloc(1,(4)*sizeof(float complex));
  J1Beam=calloc(1,(4)*sizeof(float complex));
  Unity(J0); Unity(J1);
  Unity(J0kMS); Unity(J1kMS);
  Unity(J0Beam); Unity(J1Beam);

  IMatrix=calloc(1,(4)*sizeof(float complex));
  Unity(IMatrix);


  J0inv=calloc(1,(4)*sizeof(float complex));
  J0H=calloc(1,(4)*sizeof(float complex));
  J0Conj=calloc(1,(4)*sizeof(float complex));
  J1Conj=calloc(1,(4)*sizeof(float complex));
  J1H=calloc(1,(4)*sizeof(float complex));
  J1T=calloc(1,(4)*sizeof(float complex));
  J1Hinv=calloc(1,(4)*sizeof(float complex));
  JJ=calloc(1,(4)*sizeof(float complex));
  J0kMS_tp1=calloc(1,(4)*sizeof(float complex));
  J1kMS_tp1=calloc(1,(4)*sizeof(float complex));
  J0kMS_tm1=calloc(1,(4)*sizeof(float complex));
  J1kMS_tm1=calloc(1,(4)*sizeof(float complex));
}

int JonesType;
double WaveLengthMean;
float WeightVaryJJ;
int Has_AlphaReg_killMS=0;

void initJonesServer(PyObject *LJones, int JonesTypeIn, double WaveLengthMeanIn){
  initJonesMatrices();
  WaveLengthMean=WaveLengthMeanIn;
  JonesType=JonesTypeIn;
  DoApplyJones=0;
  int LengthJonesList=PyList_Size(LJones);
  if(LengthJonesList>0){
    DoApplyJones=1;
    
    // (nt,nd,na,1,2,2)
    
    int idList=0;
    npJonesMatrices = (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrJonesMatrices=p_complex64(npJonesMatrices);
    nt_Jones=(int)npJonesMatrices->dimensions[0];
    nd_Jones=(int)npJonesMatrices->dimensions[1];
    na_Jones=(int)npJonesMatrices->dimensions[2];
    nch_Jones=(int)npJonesMatrices->dimensions[3];
    JonesDims[0]=nt_Jones;
    JonesDims[1]=nd_Jones;
    JonesDims[2]=na_Jones;
    JonesDims[3]=nch_Jones;
    npTimeMappingJonesMatrices  = (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrTimeMappingJonesMatrices = p_int32(npTimeMappingJonesMatrices);
    int size_JoneskillMS=JonesDims[0]*JonesDims[1]*JonesDims[2]*JonesDims[3];
    if(size_JoneskillMS!=0){ApplyJones_killMS=1;}
    //printf("%i, %i, %i, %i\n",JonesDims[0],JonesDims[1],JonesDims[2],JonesDims[3]);
    
    npJonesMatrices_Beam = (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrJonesMatrices_Beam=p_complex64(npJonesMatrices_Beam);
    nt_Jones_Beam=(int)npJonesMatrices_Beam->dimensions[0];
    nd_Jones_Beam=(int)npJonesMatrices_Beam->dimensions[1];
    na_Jones_Beam=(int)npJonesMatrices_Beam->dimensions[2];
    nch_Jones_Beam=(int)npJonesMatrices_Beam->dimensions[3];
    JonesDims_Beam[0]=nt_Jones_Beam;
    JonesDims_Beam[1]=nd_Jones_Beam;
    JonesDims_Beam[2]=na_Jones_Beam;
    JonesDims_Beam[3]=nch_Jones_Beam;
    npTimeMappingJonesMatrices_Beam  = (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrTimeMappingJonesMatrices_Beam = p_int32(npTimeMappingJonesMatrices_Beam);
    int size_JonesBeam=JonesDims_Beam[0]*JonesDims_Beam[1]*JonesDims_Beam[2]*JonesDims_Beam[3];
    if(size_JonesBeam!=0){ApplyJones_Beam=1;}
    //printf("%i, %i, %i, %i\n",JonesDims_Beam[0],JonesDims_Beam[1],JonesDims_Beam[2],JonesDims_Beam[3]);
    
    npA0 = (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrA0 = p_int32(npA0);
    int ifor;
    npA1= (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrA1=p_int32(npA1);

    npJonesIDIR= (PyArrayObject *) (PyList_GetItem(LJones, idList)); idList+=1;
    ptrJonesIDIR=p_int32(npJonesIDIR);
    i_dir_kMS=ptrJonesIDIR[0];

    /* if(i_dir_kMS==0){ */
    /*   int v; */
      
    /*   if (sem_init(&count_sem, 0, 1) == -1) */
    /* 	{ printf("sem_init: failed: %s\n", strerror(errno)); } */
      
    /*   // Mac OS X does not actually implement sem_getvalue() */
    /*   if (sem_getvalue(&count_sem, &v) == -1) */
    /* 	{ printf("sem_getvalue: failed: %s\n", strerror(errno)); } */
    /*   else */
    /* 	{ printf("main: count_sem value = %d\n", v); } */
    /* } */

    //i_dir_kMS=0;
    //printf("%i\n",i_dir);
    npCoefsInterp= (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrCoefsInterp=p_float32(npCoefsInterp);
    
    npJonesIDIR_Beam= (PyArrayObject *) (PyList_GetItem(LJones, idList)); idList+=1;
    ptrJonesIDIR_Beam=p_int32(npJonesIDIR_Beam);
    i_dir_Beam=ptrJonesIDIR_Beam[0];
    
    
    npModeInterpolation= (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrModeInterpolation=p_int32(npModeInterpolation);
    ModeInterpolation=ptrModeInterpolation[0];
    
    
    npVisToJonesChanMapping_killMS= (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrVisToJonesChanMapping_killMS=p_int32(npVisToJonesChanMapping_killMS);
    
    npVisToJonesChanMapping_Beam= (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrVisToJonesChanMapping_Beam=p_int32(npVisToJonesChanMapping_Beam);
    
    npAlphaReg_killMS= (PyArrayObject *) PyList_GetItem(LJones, idList); idList+=1;
    ptrAlphaReg_killMS=p_float32(npAlphaReg_killMS);
    nd_AlphaReg=(int)npAlphaReg_killMS->dimensions[0];
    na_AlphaReg=(int)npAlphaReg_killMS->dimensions[1];
    Has_AlphaReg_killMS=( nd_AlphaReg>0 );
    

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////


    PyObject *_FApplyAmp  = PyList_GetItem(LJones, idList); idList+=1;
    ApplyAmp=(int) PyFloat_AsDouble(_FApplyAmp);
    PyObject *_FApplyPhase  = PyList_GetItem(LJones, idList); idList+=1;
    ApplyPhase=(int) PyFloat_AsDouble(_FApplyPhase);
    
    PyObject *_FDoScaleJones  = PyList_GetItem(LJones, idList); idList+=1;
    DoScaleJones=(int) PyFloat_AsDouble(_FDoScaleJones);
    PyObject *_FCalibError  = PyList_GetItem(LJones, idList); idList+=1;
    CalibError=(float) PyFloat_AsDouble(_FCalibError);
    CalibError2=CalibError*CalibError;
    
    ptrSumJones=p_float64((PyArrayObject *) PyList_GetItem(LJones, idList)); idList+=1;
    ptrSumJonesChan=p_float64((PyArrayObject *) PyList_GetItem(LJones, idList)); idList+=1;
    
    PyObject *_FReWeightSNR  = PyList_GetItem(LJones, idList); idList+=1;
    ReWeightSNR=(float) PyFloat_AsDouble(_FReWeightSNR);
    
    
  };
}

int i_ant0;
int i_ant1;
float BB;

int CurrentJones_kMS_Time=-1;
int CurrentJones_kMS_Chan=-1;
int CurrentJones_Beam_Time=-1;
int CurrentJones_Beam_Chan=-1;
int CurrentJones_Index=-1;
int SameAsBefore_Beam=1;
int SameAsBefore_kMS=1;

void resetJonesServerCounter(){
  CurrentJones_Beam_Time=-1;
  CurrentJones_Beam_Chan=-1;
  CurrentJones_kMS_Time=-1;
  CurrentJones_kMS_Chan=-1;
  CurrentJones_Index=-1;
  SameAsBefore_Beam=0;
  SameAsBefore_kMS=0;
}


int DoApplyAlphaReg=0;


int SameJonesAsCurrent(int irow, int visChan){
  if((ApplyJones_Beam)&(ApplyJones_killMS)){
    int i_t=ptrTimeMappingJonesMatrices_Beam[irow];
    int i_JonesChan=ptrVisToJonesChanMapping_Beam[visChan];
    SameAsBefore_Beam=(CurrentJones_Beam_Time==i_t)&(CurrentJones_Beam_Chan=i_JonesChan);
    i_t=ptrTimeMappingJonesMatrices[irow];
    i_JonesChan=ptrVisToJonesChanMapping_killMS[visChan];
    SameAsBefore_kMS=(CurrentJones_kMS_Time==i_t)&(CurrentJones_kMS_Chan=i_JonesChan);
    return (SameAsBefore_Beam)&(SameAsBefore_kMS);
  }

  if((ApplyJones_Beam)&(!(ApplyJones_killMS))){
    int i_t=ptrTimeMappingJonesMatrices_Beam[irow];
    int i_JonesChan=ptrVisToJonesChanMapping_Beam[visChan];
    SameAsBefore_Beam=(CurrentJones_Beam_Time==i_t)&(CurrentJones_Beam_Chan=i_JonesChan);
    return SameAsBefore_Beam;
  }

  if(!(ApplyJones_Beam)){
    int i_t=ptrTimeMappingJonesMatrices[irow];
    int i_JonesChan=ptrVisToJonesChanMapping_killMS[visChan];
    SameAsBefore_kMS=(CurrentJones_kMS_Time==i_t)&(CurrentJones_kMS_Chan=i_JonesChan);
    return (SameAsBefore_kMS);
  }
}

void updateJones(int irow, int visChan, double *uvwPtr, int EstimateWeight, int DoApplyAlphaRegIn){

  if(SameJonesAsCurrent(irow, visChan)){return;}


  i_ant0=ptrA0[irow];
  i_ant1=ptrA1[irow];
  DoApplyAlphaReg=0;
  if(DoApplyAlphaRegIn & Has_AlphaReg_killMS){DoApplyAlphaReg=1;};
  DoApplyAlphaReg=0;


  //printf("(%i, %i)\n",i_ant0,i_ant1);
  //int JonesChannel_Beam=ptrVisToJonesChanMapping_Beam


  int SomeJonesHaveChanged=0;

  if(ApplyJones_Beam){
    int i_t=ptrTimeMappingJonesMatrices_Beam[irow];
    int i_JonesChan=ptrVisToJonesChanMapping_Beam[visChan];
    //printf("ptrVisToJonesChanMapping_Beam[visChan]=%i %i\n;",visChan,ptrVisToJonesChanMapping_Beam[visChan]);
    SameAsBefore_Beam=(CurrentJones_Beam_Time==i_t)&(CurrentJones_Beam_Chan==i_JonesChan);

    if(SameAsBefore_Beam==0){
      GiveJones(ptrJonesMatrices_Beam, JonesDims_Beam, ptrCoefsInterp, i_t, i_ant0, i_dir_Beam, i_JonesChan, ModeInterpolation, J0Beam);
      GiveJones(ptrJonesMatrices_Beam, JonesDims_Beam, ptrCoefsInterp, i_t, i_ant1, i_dir_Beam, i_JonesChan, ModeInterpolation, J1Beam);
      CurrentJones_Beam_Time=i_t;
      CurrentJones_Beam_Chan=i_JonesChan;
      SomeJonesHaveChanged=1;
    }

  }


  //ApplyJones_killMS=1;
  if(ApplyJones_killMS){
    int i_t=ptrTimeMappingJonesMatrices[irow];
    int i_JonesChan=ptrVisToJonesChanMapping_killMS[visChan];
    SameAsBefore_kMS=(CurrentJones_kMS_Time==i_t)&(CurrentJones_kMS_Chan==i_JonesChan);
    
    //SameAsBefore_kMS=0;

    if(SameAsBefore_kMS==0){
      GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant0, i_dir_kMS, i_JonesChan, ModeInterpolation, J0kMS);
      GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant1, i_dir_kMS, i_JonesChan, ModeInterpolation, J1kMS);
      if(DoApplyAlphaReg){
	size_t off_alpha0=i_dir_kMS*na_AlphaReg+i_ant0;
	size_t off_alpha1=i_dir_kMS*na_AlphaReg+i_ant1;
	float alpha0=*(ptrAlphaReg_killMS+off_alpha0);
	float alpha1=*(ptrAlphaReg_killMS+off_alpha1);
	int ipol;
	//alpha0=0.;
	//alpha1=0.;

	//printf("akpha0=%f\n",alpha0);
	//printf("akpha1=%f\n",alpha1);
	for(ipol=0;ipol<4;ipol++){
	  J0kMS[ipol]=J0kMS[ipol]*alpha0+(1.-alpha0)*IMatrix[ipol];
	  J1kMS[ipol]=J1kMS[ipol]*alpha1+(1.-alpha1)*IMatrix[ipol];
	}
      }


      NormJones(J0kMS, ApplyAmp, ApplyPhase, DoScaleJones, uvwPtr, WaveLengthMean, CalibError);
      NormJones(J1kMS, ApplyAmp, ApplyPhase, DoScaleJones, uvwPtr, WaveLengthMean, CalibError);
      CurrentJones_kMS_Time=i_t;
      CurrentJones_kMS_Chan=i_JonesChan;
      SomeJonesHaveChanged=1;
    
      WeightVaryJJ=1.;

      if((EstimateWeight==1)&(ReWeightSNR!=0)){
	int i_t_p1;
	i_t_p1=i_t+1;
	if (i_t==(nt_Jones-1)){i_t_p1=i_t;}
	GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t_p1, i_ant0, i_dir_kMS, i_JonesChan, ModeInterpolation, J0kMS_tp1);
	GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t_p1, i_ant1, i_dir_kMS, i_JonesChan, ModeInterpolation, J1kMS_tp1);
	/* float abs_dg0=cabs(J0kMS_tp1[0]-J0kMS[0]); */
	/* float abs_dg1=cabs(J1kMS_tp1[0]-J1kMS[0]); */
	
	int i_t_m1;
	i_t_m1=i_t-1;
	if (i_t==0){i_t_m1=i_t;}
	GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t_m1, i_ant0, i_dir_kMS, i_JonesChan, ModeInterpolation, J0kMS_tm1);
	GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t_m1, i_ant1, i_dir_kMS, i_JonesChan, ModeInterpolation, J1kMS_tm1);
	float abs_dg0=cabs(J0kMS_tp1[0]-J0kMS[0])+cabs(J0kMS_tm1[0]-J0kMS[0]);
	float abs_dg1=cabs(J1kMS_tp1[0]-J1kMS[0])+cabs(J1kMS_tm1[0]-J1kMS[0]);
	
	
	
	
	float abs_g0=cabs(J0kMS[0]);
	float abs_g1=cabs(J1kMS[0]);
	
	//float V0=abs_dg0*abs_dg1;
	//WeightVaryJJ=1./(1.+V0*V0);
	
	/* WeightVaryJJ  = 1./((1.+abs_dg0)*(1.+abs_dg1)); */
	/* WeightVaryJJ *= WeightVaryJJ; */
	
	
	// Works well
	/* float Ang0=cargf(J0kMS[0]); */
	/* float Ang1=cargf(J1kMS[0]); */
	/* float Rij_Ion=abs_g0*abs_g1*cabs(cexp(I*(Ang1-Ang0)*FracFreqWidth/2.)-1.); */
	/* float Rij=(Rij_Ion+abs_g1*abs_dg0+abs_g0*abs_dg1)*ReWeightSNR; */
	
	float Rij=(abs_g1*abs_dg0+abs_g0*abs_dg1)*ReWeightSNR;
	//float Rij=(abs_g1*abs_dg0*abs_g1*abs_dg0+abs_g0*abs_dg1*abs_g0*abs_dg1)*ReWeightSNR;
	
	float Rij_sq=1.+Rij*Rij;
	WeightVaryJJ  = 1./(Rij_sq);
	
	float abs_g0_3=cabs(J0kMS[3]);
	float abs_g1_3=cabs(J1kMS[3]);
	if( ((abs_g0*abs_g1)>2.) | ((abs_g0_3*abs_g1_3)>2.) ){WeightVaryJJ=0.;};
	
	/* // TEST */
	/* float Rij=(abs_g1*abs_dg0+abs_g0*abs_dg1)*ReWeightSNR; */
	/* float Rij_sq=1.+Rij*Rij; */
	/* float V0=0.01; */
	/* float Vg0=V0+abs_dg0*abs_dg0+abs_g0*abs_g0; */
	/* float Vg1=V0+abs_dg1*abs_dg1+abs_g1*abs_g1; */
	/* float Vgigj=Vg0*Vg1; */
	/* float V=Vgigj*Rij_sq; */
	/* WeightVaryJJ  = 1./(V); */
      }
      
    }

    

    
  }
  




  if(SomeJonesHaveChanged){
    CurrentJones_Index+=1;
    J0[0]=1;J0[1]=0;J0[2]=0;J0[3]=1;
    J1[0]=1;J1[1]=0;J1[2]=0;J1[3]=1;
    if(ApplyJones_Beam){
      MatDot(J0Beam,JonesType,J0,JonesType,J0);
      MatDot(J1Beam,JonesType,J1,JonesType,J1);
    }
    
    if(ApplyJones_killMS){
      MatDot(J0kMS,JonesType,J0,JonesType,J0);
      MatDot(J1kMS,JonesType,J1,JonesType,J1);
    }
    
    MatT(J1,J1T);
    MatConj(J0,J0Conj);
    //MatConj(J1,J1Conj);
    //BB=cabs(J0Conj[0]*J1T[0]);
    //BB=cabs(J0Conj[0]*J0[0]*J1[0]*J1Conj[0]);
    //BB*=BB;
    BB=(cabs(J0[0])*cabs(J1[0])+cabs(J0[3])*cabs(J1[3]))/2.;
    BB*=BB;
    MatH(J1,J1H);
    MatH(J0,J0H);
  }
  /* MatInv(J0,J0inv,0); */
  /* MatH(J1,J1H); */
  /* MatInv(J1H,J1Hinv,0); */
  /* MatDot(J0inv,JonesType,J1Hinv,JonesType,JJ); */
  /* BB=cabs(JJ[0]); */
  
  /* //BB*=BB; */
}
