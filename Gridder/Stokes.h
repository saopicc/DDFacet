/**
 * This library simply wraps casacore's c++ stokes converter to a c conversion routine.
 */

#pragma once
#ifdef __cplusplus
  #include <casacore/measures/Measures/Stokes.h>
  #include <casacore/ms/MeasurementSets/StokesConverter.h>
  #include <casacore/casa/vector.h>
  #include <casacore/casa/complex.h>
#endif

#include <stdint.h>
#include <complex.h>
#ifdef __cplusplus
extern "C" {
#endif
  // Converts float32 complex visibility from one correlation format to
  // another (with the format specified by casacore's Stokes.h).
  // This method can be used to convert from correlations to stokes
  // parameters or the other way around.
  // Inputs:
  // 	in_format_len, out_format_len: length of the the arrays
  //		describing the input and desired output layouts of the
  //		visibility to be converted.
  //	in_format, out_format: arrays containing formats.
  //	in_data: input visibility with the length
  //		 in_format_len (usually a 4 correlation term).
  //	out_data: output visibility with the length
  //		out_format_len.
  
  void convert_corrs_32(size_t in_format_len, 
		      	size_t out_format_len,
		      	int * in_format,
		      	int * out_format,
			float _Complex * in_data,
			float _Complex * out_data);
  
  // Converts float64 complex visibility from one correlation format to
  // another (with the format specified by casacore's Stokes.h).
  // This method can be used to convert from correlations to stokes
  // parameters or the other way around. See also convert_corrs_32.
  void convert_corrs_64(size_t in_format_len, 
		      	size_t out_format_len,
		      	int * in_format,
		      	int * out_format,
			double _Complex * in_data,
			double _Complex * out_data);

#ifdef __cplusplus
}
#endif

