#include "Stokes.h"
#include <complex>
  void convert_corrs_32(size_t in_format_len, 
		      	size_t out_format_len,
		      	int * in_format,
		      	int * out_format,
			float _Complex * in_data,
			float _Complex * out_data){
  casa::Vector<casa::Int> out(out_format_len);
  casa::Vector<casa::Int> in(in_format_len);
  casacore::Vector<casacore::Complex> casa_data_in(in_format_len);
  casacore::Vector<casacore::Complex> casa_data_out(out_format_len);
  for (std::size_t corr = 0; corr < out_format_len; ++corr){
   out[corr] = static_cast<casa::Int>(out_format[corr]);
  }
  for (std::size_t corr = 0; corr < in_format_len; ++corr){
   in[corr] = static_cast<casa::Int>(in_format[corr]);
   casa_data_in[corr] = casacore::Complex(creal(in_data[corr]),cimag(in_data[corr]));
  }
  casacore::StokesConverter conv(out,in,true);
  conv.convert(casa_data_out,casa_data_in);
  for (std::size_t corr = 0; corr < out_format_len; ++corr){
    out_data[corr] = static_cast<float>(casa_data_out[corr].real()) + static_cast<float>(casa_data_out[corr].imag()) * I;
  }
}
  void convert_corrs_64(size_t in_format_len, 
		      	size_t out_format_len,
		      	int * in_format,
		      	int * out_format,
			double _Complex * in_data,
			double _Complex * out_data){
  casa::Vector<casa::Int> out(out_format_len);
  casa::Vector<casa::Int> in(in_format_len);
  casacore::Vector<casacore::Complex> casa_data_in(in_format_len);
  casacore::Vector<casacore::Complex> casa_data_out(out_format_len);
  for (std::size_t corr = 0; corr < out_format_len; ++corr){
   out[corr] = static_cast<casa::Int>(out_format[corr]);
  }
  for (std::size_t corr = 0; corr < in_format_len; ++corr){
   in[corr] = static_cast<casa::Int>(in_format[corr]);
   casa_data_in[corr] = casacore::Complex(creal(in_data[corr]),cimag(in_data[corr]));
  }
  casacore::StokesConverter conv(out,in,true);
  conv.convert(casa_data_out,casa_data_in);
  for (std::size_t corr = 0; corr < out_format_len; ++corr){
    out_data[corr] = static_cast<double>(casa_data_out[corr].real()) + static_cast<double>(casa_data_out[corr].imag()) * I;
  }
}