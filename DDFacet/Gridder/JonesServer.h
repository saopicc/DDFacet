template<typename T> class Mat
  {
  private:
    T v[4];

  public:
    Mat() {}
    Mat(T v0, T v1, T v2, T v3)
      : v { v0, v1, v2, v3 } {}
    T &operator[](size_t i) { return v[i]; }
    const T &operator[](size_t i) const { return v[i]; }
    void setZero()
      { v[0]=v[1]=v[2]=v[3]=T(0); }
    void setUnity()
      {
      v[0]=v[3]=T(1);
      v[1]=v[2]=T(0);
      }
    void set(T v0, T v1, T v2, T v3)
      { v = { v0, v1, v2, v3 }; }
    Mat hermitian() const
      { return Mat(conj(v[0]), conj(v[2]), conj(v[1]), conj(v[3])); }
    void scale(T lambda)
      { v[0]*=lambda; v[1]*=lambda; v[2]*=lambda; v[3]*=lambda; }
    Mat times(const Mat &b) const
      {
      return Mat(v[0]*b.v[0]+v[1]*b.v[2],
                 v[0]*b.v[1]+v[1]*b.v[3],
                 v[2]*b.v[0]+v[3]*b.v[2],
                 v[2]*b.v[1]+v[3]*b.v[3]);
      }
  };

using dcMat = Mat<dcmplx>;

class JonesServer
  {
  private:
    void NormJones(dcMat &J0, const double *uvwPtr) const
      {
      if (!ApplyAmp)
        for (int ThisPol=0; ThisPol<4; ThisPol++)
          {
          double aj0 = abs(J0[ThisPol]);
          if(aj0!=0.)
            J0[ThisPol]/=aj0;
          }

      if (!ApplyPhase)
        for(int ThisPol=0; ThisPol<4;ThisPol++)
          J0[ThisPol]=abs(J0[ThisPol]);

      if (DoScaleJones)
        {
        double U2=uvwPtr[0]*uvwPtr[0];
        double V2=uvwPtr[1]*uvwPtr[1];
        double R2=(U2+V2)/(WaveLengthMean*WaveLengthMean);
        double AlphaScaleJones=exp(-2.*PI*CalibError*CalibError*R2);
        for (int ThisPol=0; ThisPol<4; ThisPol++)
          {
          double aj0 = abs(J0[ThisPol]);
          if(aj0!=0.)
            J0[ThisPol] *= (1.-AlphaScaleJones)/aj0 + AlphaScaleJones;
          }
        }
      }

    static dcMat GiveJones(const fcmplx *ptrJonesMatrices, const int *JonesDims,
      const float *ptrCoefs, int i_t, int i_ant0, int i_dir, int iChJones,
      int Mode)
      {
      size_t nd_Jones =size_t(JonesDims[1]),
             na_Jones =size_t(JonesDims[2]),
             nch_Jones=size_t(JonesDims[3]);
      size_t offJ0=size_t(i_t)*nd_Jones*na_Jones*nch_Jones*4
                  +i_dir*na_Jones*nch_Jones*4
                  +i_ant0*nch_Jones*4
                  +iChJones*4;
      dcMat Jout;

      if (Mode==0)
        for (auto ipol=0; ipol<4; ipol++)
          Jout[ipol]=ptrJonesMatrices[offJ0+ipol];

      else if (Mode==1)
        {
        double Jabs[4]={0,0,0,0};
        Jout.setZero();

        // MR FIXME: swap loops?
        for (size_t idir=0; idir<nd_Jones; idir++)
          {
          double coeff = ptrCoefs[idir];
          if (coeff==0) continue;

          for (auto ipol=0; ipol<4; ipol++)
            {
            dcmplx val = ptrJonesMatrices[offJ0+ipol];
            double A = abs(val);
            Jout[ipol]+=coeff/A*val;
            Jabs[ipol]+=coeff*A;
            }
          }

        for (auto ipol=0; ipol<4; ipol++)
          Jout[ipol]*=Jabs[ipol];
        }
      return Jout;
      }

    const fcmplx *ptrJonesMatrices, *ptrJonesMatrices_Beam;
    const int *ptrTimeMappingJonesMatrices, *ptrTimeMappingJonesMatrices_Beam,
              *ptrVisToJonesChanMapping_killMS,*ptrVisToJonesChanMapping_Beam,
              *ptrA0, *ptrA1;
    const float *ptrCoefsInterp;
    int i_dir_kMS;

    const float *ptrAlphaReg_killMS;

    int na_AlphaReg;
    int JonesDims_Beam[4];
    bool ApplyJones_Beam=false;
    int i_dir_Beam;
    bool ApplyJones_killMS=false;
    int nt_Jones;

    int JonesDims[4];
    int ModeInterpolation=1;
    bool ApplyAmp, ApplyPhase, DoScaleJones;
    double CalibError, ReWeightSNR;

    double WaveLengthMean;
    bool Has_AlphaReg_killMS=false;

    int CurrentJones_kMS_Time=-1;
    int CurrentJones_kMS_Chan=-1;
    int CurrentJones_Beam_Time=-1;
    int CurrentJones_Beam_Chan=-1;

  public:
    JonesServer(PyObject *LJones, double WaveLengthMeanIn){
      J0.setUnity(); J1.setUnity();
      WaveLengthMean=WaveLengthMeanIn;
      DoApplyJones=(PyList_Size(LJones)>0);
      if (DoApplyJones)
        {
        PyArrayObject *npJonesMatrices = (PyArrayObject *) PyList_GetItem(LJones, 0);
        ptrJonesMatrices=p_complex64(npJonesMatrices);
        JonesDims[0]=nt_Jones=(int)npJonesMatrices->dimensions[0];
        JonesDims[1]=(int)npJonesMatrices->dimensions[1];
        JonesDims[2]=(int)npJonesMatrices->dimensions[2];
        JonesDims[3]=(int)npJonesMatrices->dimensions[3];
        ptrTimeMappingJonesMatrices = p_int32((PyArrayObject *)PyList_GetItem(LJones, 1));
        ApplyJones_killMS=(JonesDims[0]*JonesDims[1]*JonesDims[2]*JonesDims[3]!=0);

        PyArrayObject *npJonesMatrices_Beam = (PyArrayObject *) PyList_GetItem(LJones, 2);
        ptrJonesMatrices_Beam=p_complex64(npJonesMatrices_Beam);
        JonesDims_Beam[0]=(int)npJonesMatrices_Beam->dimensions[0];
        JonesDims_Beam[1]=(int)npJonesMatrices_Beam->dimensions[1];
        JonesDims_Beam[2]=(int)npJonesMatrices_Beam->dimensions[2];
        JonesDims_Beam[3]=(int)npJonesMatrices_Beam->dimensions[3];
        PyArrayObject *npTimeMappingJonesMatrices_Beam  = (PyArrayObject *) PyList_GetItem(LJones, 3);
        ptrTimeMappingJonesMatrices_Beam = p_int32(npTimeMappingJonesMatrices_Beam);
        ApplyJones_Beam=(JonesDims_Beam[0]*JonesDims_Beam[1]*JonesDims_Beam[2]*JonesDims_Beam[3]!=0);

        ptrA0 = p_int32((PyArrayObject *) PyList_GetItem(LJones, 4));
        ptrA1=p_int32((PyArrayObject *) PyList_GetItem(LJones, 5));

        i_dir_kMS=p_int32((PyArrayObject *) (PyList_GetItem(LJones, 6)))[0];

        ptrCoefsInterp=p_float32((PyArrayObject *) PyList_GetItem(LJones, 7));

        i_dir_Beam=p_int32((PyArrayObject *) (PyList_GetItem(LJones, 8)))[0];

        ModeInterpolation=p_int32((PyArrayObject *) PyList_GetItem(LJones, 9))[0];

        ptrVisToJonesChanMapping_killMS=p_int32((PyArrayObject *) PyList_GetItem(LJones, 10));

        ptrVisToJonesChanMapping_Beam=p_int32((PyArrayObject *) PyList_GetItem(LJones, 11));

        PyArrayObject *npAlphaReg_killMS= (PyArrayObject *) PyList_GetItem(LJones, 12);
        ptrAlphaReg_killMS=p_float32(npAlphaReg_killMS);
        Has_AlphaReg_killMS=(npAlphaReg_killMS->dimensions[0]>0);
        na_AlphaReg=int(npAlphaReg_killMS->dimensions[1]);

        ApplyAmp=bool(PyFloat_AsDouble(PyList_GetItem(LJones, 13)));
        ApplyPhase=bool(PyFloat_AsDouble(PyList_GetItem(LJones, 14)));

        DoScaleJones=bool(PyFloat_AsDouble(PyList_GetItem(LJones, 15)));
        CalibError=PyFloat_AsDouble(PyList_GetItem(LJones, 16));

        ptrSumJones=p_float64((PyArrayObject *) PyList_GetItem(LJones, 17));
        ptrSumJonesChan=p_float64((PyArrayObject *) PyList_GetItem(LJones, 18));

        ReWeightSNR=PyFloat_AsDouble(PyList_GetItem(LJones, 19));
        }
      }

    void updateJones(size_t irow, size_t visChan, const double *uvwPtr, bool EstimateWeight, bool DoApplyAlphaRegIn){
      int i_ant0=ptrA0[irow];
      int i_ant1=ptrA1[irow];
      // MR FIXME
      bool DoApplyAlphaReg = (DoApplyAlphaRegIn && Has_AlphaReg_killMS);
      DoApplyAlphaReg=false;

      if (ApplyJones_Beam&&ApplyJones_killMS){
        int i_t=ptrTimeMappingJonesMatrices_Beam[irow];
        int i_JonesChan=ptrVisToJonesChanMapping_Beam[visChan];
        bool SameAsBefore_Beam=(CurrentJones_Beam_Time==i_t)&&(CurrentJones_Beam_Chan==i_JonesChan);
        i_t=ptrTimeMappingJonesMatrices[irow];
        i_JonesChan=ptrVisToJonesChanMapping_killMS[visChan];
        bool SameAsBefore_kMS=(CurrentJones_kMS_Time==i_t)&&(CurrentJones_kMS_Chan==i_JonesChan);
        if(SameAsBefore_Beam&&SameAsBefore_kMS) return;
        }

      bool SomeJonesHaveChanged=false;
      dcMat J0Beam(1,0,0,1), J1Beam(1,0,0,1);

      if (ApplyJones_Beam){
        int i_t=ptrTimeMappingJonesMatrices_Beam[irow];
        int i_JonesChan=ptrVisToJonesChanMapping_Beam[visChan];
        bool SameAsBefore_Beam=(CurrentJones_Beam_Time==i_t)&&(CurrentJones_Beam_Chan==i_JonesChan);

        if (!SameAsBefore_Beam){
          J0Beam = GiveJones(ptrJonesMatrices_Beam, JonesDims_Beam, ptrCoefsInterp, i_t, i_ant0, i_dir_Beam, i_JonesChan, ModeInterpolation);
          J1Beam = GiveJones(ptrJonesMatrices_Beam, JonesDims_Beam, ptrCoefsInterp, i_t, i_ant1, i_dir_Beam, i_JonesChan, ModeInterpolation);
          CurrentJones_Beam_Time=i_t;
          CurrentJones_Beam_Chan=i_JonesChan;
          SomeJonesHaveChanged=true;
          }
        }

      dcMat J0kMS(1,0,0,1), J1kMS(1,0,0,1);
      if (ApplyJones_killMS){
        int i_t=ptrTimeMappingJonesMatrices[irow];
        int i_JonesChan=ptrVisToJonesChanMapping_killMS[visChan];
        bool SameAsBefore_kMS=(CurrentJones_kMS_Time==i_t)&&(CurrentJones_kMS_Chan==i_JonesChan);

        if(!SameAsBefore_kMS){
          J0kMS = GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant0, i_dir_kMS, i_JonesChan, ModeInterpolation);
          J1kMS = GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant1, i_dir_kMS, i_JonesChan, ModeInterpolation);
          if(DoApplyAlphaReg){
            size_t off_alpha0=i_dir_kMS*na_AlphaReg+i_ant0;
            size_t off_alpha1=i_dir_kMS*na_AlphaReg+i_ant1;
            double alpha0=ptrAlphaReg_killMS[off_alpha0];
            double alpha1=ptrAlphaReg_killMS[off_alpha1];
            dcMat IMatrix(1,0,0,1);

            for(int ipol=0;ipol<4;ipol++){
              J0kMS[ipol]=J0kMS[ipol]*alpha0+(1-alpha0)*IMatrix[ipol];
              J1kMS[ipol]=J1kMS[ipol]*alpha1+(1-alpha1)*IMatrix[ipol];
              }
            }

          NormJones(J0kMS, uvwPtr);
          NormJones(J1kMS, uvwPtr);
          CurrentJones_kMS_Time=i_t;
          CurrentJones_kMS_Chan=i_JonesChan;
          SomeJonesHaveChanged=true;

          if (EstimateWeight){
            int i_t_p1=i_t+1;
            if (i_t==nt_Jones-1) i_t_p1=i_t;
            dcMat J0kMS_tp1 = GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t_p1, i_ant0, i_dir_kMS, i_JonesChan, ModeInterpolation);
            dcMat J1kMS_tp1 = GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t_p1, i_ant1, i_dir_kMS, i_JonesChan, ModeInterpolation);

            int i_t_m1=i_t-1;
            if (i_t==0) i_t_m1=i_t;
            dcMat J0kMS_tm1 = GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t_m1, i_ant0, i_dir_kMS, i_JonesChan, ModeInterpolation);
            dcMat J1kMS_tm1 = GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t_m1, i_ant1, i_dir_kMS, i_JonesChan, ModeInterpolation);
            double abs_dg0=abs(J0kMS_tp1[0]-J0kMS[0])+abs(J0kMS_tm1[0]-J0kMS[0]);
            double abs_dg1=abs(J1kMS_tp1[0]-J1kMS[0])+abs(J1kMS_tm1[0]-J1kMS[0]);

            double abs_g0=abs(J0kMS[0]);
            double abs_g1=abs(J1kMS[0]);

            double Rij=(abs_g1*abs_dg0+abs_g0*abs_dg1)*ReWeightSNR;
            WeightVaryJJ  = 1./(1.+Rij*Rij);

            double abs_g0_3=abs(J0kMS[3]);
            double abs_g1_3=abs(J1kMS[3]);
            if ((abs_g0*abs_g1>2.) || (abs_g0_3*abs_g1_3>2.)) WeightVaryJJ=0.;
            }
          }
        }

      if (SomeJonesHaveChanged)
        {
        J0.setUnity();
        J1.setUnity();
        if (ApplyJones_Beam)
          {
          J0=J0Beam.times(J0);
          J1=J1Beam.times(J1);
          }
        if (ApplyJones_killMS)
          {
          J0=J0kMS.times(J0);
          J1=J1kMS.times(J1);
          }

        BB=(abs(J0[0])*abs(J1[0])+abs(J0[3])*abs(J1[3]))/2.;
        BB*=BB;
        J0H=J0.hermitian();
        J1H=J1.hermitian();
        }
      }

    void resetJonesServerCounter()
      {
      CurrentJones_Beam_Time=CurrentJones_Beam_Chan=-1;
      CurrentJones_kMS_Time=CurrentJones_kMS_Chan=-1;
      }
    double BB;
    double WeightVaryJJ;
    bool DoApplyJones=false;
    dcMat J0, J1, J0H, J1H;
    double *ptrSumJones, *ptrSumJonesChan;
  };
