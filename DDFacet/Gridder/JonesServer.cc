/**
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
*/

#include "JonesServer.h"
#include <iostream>
namespace DDF {
  namespace DDEs {
    void JonesServer::NormJones(dcMat &J0, const double *uvwPtr) const
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

    dcMat JonesServer::GiveJones(const fcmplx *ptrJonesMatrices, const int *JonesDims,
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

    JonesServer::JonesServer(py::list& LJones, double WaveLengthMeanIn){
      J0Beam.setUnity();
      J1Beam.setUnity();
      J0kMS.setUnity();
      J1kMS.setUnity();
      J0.setUnity(); J1.setUnity();
      WaveLengthMean=WaveLengthMeanIn;
      DoApplyJones=int(LJones.size());

      if (DoApplyJones)
	{
	// KillMS solutions
	//BH FIXME: needs comment and check on the layout of the solutions
	auto npJonesMatrices = py::array_t<std::complex<float>, py::array::c_style>(LJones[0]);
	if (npJonesMatrices.ndim() == 6 &&
	    npJonesMatrices.size() != 0) {
	  ptrJonesMatrices=npJonesMatrices.data(0);
	  JonesDims[0]=nt_Jones=int(npJonesMatrices.shape(0));
	  JonesDims[1]=int(npJonesMatrices.shape(1));
	  JonesDims[2]=int(npJonesMatrices.shape(2));
	  JonesDims[3]=int(npJonesMatrices.shape(3));
	  auto timeMappingJonesMatricies = py::array_t<int, py::array::c_style>(LJones[1]);
	  ptrTimeMappingJonesMatrices = timeMappingJonesMatricies.size() != 0 ? timeMappingJonesMatricies.data(0) : nullptr;
	  ptrVisToJonesChanMapping_killMS=py::array_t<int, py::array::c_style>(LJones[10]).data(0);
	  i_dir_kMS=py::array_t<int32_t,py::array::c_style>(LJones[6]).data(0)[0];
	  ApplyJones_killMS = true;
	} else {
	  ptrJonesMatrices=nullptr;
	  JonesDims[0] = JonesDims[1] = JonesDims[2] = JonesDims[3] = nt_Jones = 0;
	  ApplyJones_killMS = false;
	  ptrTimeMappingJonesMatrices = nullptr;
	  ptrVisToJonesChanMapping_killMS = nullptr;
	  i_dir_kMS = 0;
	}

	//E-Jones solutions
	//BH FIXME: needs comment and check on the layout of the solutions
	auto npJonesMatrices_Beam = py::array_t<std::complex<float>, py::array::c_style>(LJones[2]);
	if (npJonesMatrices_Beam.ndim() == 6 &&
	    npJonesMatrices_Beam.size() != 0){
	  ptrJonesMatrices_Beam=npJonesMatrices_Beam.data(0);
	  JonesDims_Beam[0]=int(npJonesMatrices_Beam.shape(0));
	  JonesDims_Beam[1]=int(npJonesMatrices_Beam.shape(1));
	  JonesDims_Beam[2]=int(npJonesMatrices_Beam.shape(2));
	  JonesDims_Beam[3]=int(npJonesMatrices_Beam.shape(3));
	  auto npTimeMappingJonesMatrices_Beam  = py::array_t<int, py::array::c_style>(LJones[3]);
	  ptrTimeMappingJonesMatrices_Beam = npTimeMappingJonesMatrices_Beam.size() != 0 ? npTimeMappingJonesMatrices_Beam.data(0) : nullptr;
	  ApplyJones_Beam = true;
	  ptrVisToJonesChanMapping_Beam=py::array_t<int, py::array::c_style>(LJones[11]).data(0);
	  i_dir_Beam=py::array_t<int32_t,py::array::c_style>(LJones[8]).data(0)[0];
	} else {
	  JonesDims_Beam[0]=JonesDims_Beam[1]=JonesDims_Beam[2]=JonesDims_Beam[3]=0;
	  ptrJonesMatrices_Beam = nullptr;
	  ptrTimeMappingJonesMatrices_Beam = nullptr;
	  ApplyJones_Beam = false;
	  ptrVisToJonesChanMapping_Beam = nullptr;
	  i_dir_Beam = 0;
	}
	if (not (ApplyJones_Beam || ApplyJones_killMS))
	  throw std::runtime_error("Jones matricies specified but neither E or DD Jones are applied. This is a bug!");

	auto A0 = py::array_t<int, py::array::c_style>(LJones[4]);
	ptrA0 = A0.size() != 0 ? A0.data(0) : nullptr;
	auto A1 = py::array_t<int, py::array::c_style>(LJones[5]);
	ptrA1 = A1.size() != 0 ? A1.data(0) : nullptr;

	auto coefsInterp = py::array_t<float, py::array::c_style>(LJones[7]);
	ptrCoefsInterp = coefsInterp.size() != 0 ? coefsInterp.data(0) : nullptr;

	ModeInterpolation=py::array_t<int32_t,py::array::c_style>(LJones[9]).data(0)[0];
	DoApplyJones=py::array_t<int32_t,py::array::c_style>(LJones[9]).data(0)[1];

	auto npAlphaReg_killMS= py::array_t<float, py::array::c_style>(LJones[12]);
	if (npAlphaReg_killMS.ndim() == 2 &&
	    npAlphaReg_killMS.shape(0) > 0 && npAlphaReg_killMS.shape(1) > 0){
	  ptrAlphaReg_killMS = npAlphaReg_killMS.data(0);
	  Has_AlphaReg_killMS = (npAlphaReg_killMS.shape(0) > 0);
	  na_AlphaReg = int(npAlphaReg_killMS.shape(1));
	} else {
	  ptrAlphaReg_killMS = nullptr;
	  Has_AlphaReg_killMS = false;
	  na_AlphaReg = 0;
	}
	ApplyAmp=LJones[13].cast<bool>();
	ApplyPhase=LJones[14].cast<bool>();

	DoScaleJones=LJones[15].cast<bool>();
	CalibError=LJones[16].cast<double>();

	ptrSumJones=py::array_t<double, py::array::c_style>(LJones[17]).mutable_data(0);
	ptrSumJonesChan=py::array_t<double, py::array::c_style>(LJones[18]).mutable_data(0);

	ReWeightSNR=LJones[19].cast<double>();
	}
      }

    bool JonesServer::updateJones(size_t irow, size_t visChan, const double *uvwPtr, bool EstimateWeight, bool DoApplyAlphaRegIn){
      int i_ant0=ptrA0[irow];
      int i_ant1=ptrA1[irow];
      // MR FIXME
      bool DoApplyAlphaReg = (DoApplyAlphaRegIn && Has_AlphaReg_killMS);
      DoApplyAlphaReg=false;

      bool baseline_changed = i_ant0 != CurrentJones_ant0 || i_ant1 != CurrentJones_ant1;
      bool SomeJonesHaveChanged = false;
      CurrentJones_ant0 = i_ant0;
      CurrentJones_ant1 = i_ant1;

      if (ApplyJones_Beam)
        {
	int i_t=ptrTimeMappingJonesMatrices_Beam[irow];
	int i_JonesChan=ptrVisToJonesChanMapping_Beam[visChan];
	if( baseline_changed || CurrentJones_Beam_Time!=i_t || CurrentJones_Beam_Chan!=i_JonesChan)
	  {
	  J0Beam = GiveJones(ptrJonesMatrices_Beam, JonesDims_Beam, ptrCoefsInterp, i_t, i_ant0, i_dir_Beam, i_JonesChan, ModeInterpolation);
	  J1Beam = GiveJones(ptrJonesMatrices_Beam, JonesDims_Beam, ptrCoefsInterp, i_t, i_ant1, i_dir_Beam, i_JonesChan, ModeInterpolation);
	  CurrentJones_Beam_Time=i_t;
	  CurrentJones_Beam_Chan=i_JonesChan;
	  SomeJonesHaveChanged=true;
	  }
	}

      if (ApplyJones_killMS)
        {
	int i_t=ptrTimeMappingJonesMatrices[irow];
	int i_JonesChan=ptrVisToJonesChanMapping_killMS[visChan];
	if( baseline_changed || CurrentJones_kMS_Time!=i_t || CurrentJones_kMS_Chan!=i_JonesChan)
	{
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
	  //std::cout<< "EstimateWeight"<<EstimateWeight <<std::endl;
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
	    //std::cout<< "aa"<<WeightVaryJJ <<std::endl;
	    //cout<<"dd"<<endl;
	    if ((abs_g0*abs_g1>2.) || (abs_g0_3*abs_g1_3>2.)) {
	      //std::cout<<  "bb"<<WeightVaryJJ <<std::endl;
	      WeightVaryJJ=0.;};
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
        return true;
	}
      return false;
      }

    void JonesServer::resetJonesServerCounter()
      {
      CurrentJones_ant0=CurrentJones_ant1=-1;
      CurrentJones_Beam_Time=CurrentJones_Beam_Chan=-1;
      CurrentJones_kMS_Time=CurrentJones_kMS_Chan=-1;
      }
  }
}
