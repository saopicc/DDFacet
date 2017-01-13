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

#include "Stokes.h"

casacore::StokesConverter * _conv;
size_t _in_format_len, _out_format_len;
void init_stokes_converter(size_t in_format_len, 
			     size_t out_format_len,
			     int * in_format,
			     int * out_format)
{
    casa::Vector<casa::Int> out ( out_format_len );
    casa::Vector<casa::Int> in ( in_format_len );
    for ( std::size_t corr = 0; corr < out_format_len; ++corr ) {
        out[corr] = static_cast<casa::Int> ( out_format[corr] );
    }
    for ( std::size_t corr = 0; corr < in_format_len; ++corr ) {
        in[corr] = static_cast<casa::Int> ( in_format[corr] );
    }
    _conv = new casacore::StokesConverter( out,in,true );
    _in_format_len = in_format_len;
    _out_format_len = out_format_len;
}

void free_stokes_library()
{
  delete _conv;
}

void convert_corrs_32 (float _Complex * in_data, float _Complex * out_data )
{
    casacore::Vector<casacore::Complex> casa_data_in ( _in_format_len );
    casacore::Vector<casacore::Complex> casa_data_out ( _out_format_len );
    for ( std::size_t corr = 0; corr < _in_format_len; ++corr ) {
        casa_data_in[corr] = casacore::Complex ( creal ( in_data[corr] ),cimag ( in_data[corr] ) );
    }
    _conv->convert ( casa_data_out,casa_data_in );
    for ( std::size_t corr = 0; corr < _out_format_len; ++corr ) {
        out_data[corr] = static_cast<float> ( casa_data_out[corr].real() ) + static_cast<float> ( casa_data_out[corr].imag() ) * _Complex_I;
    }
}

void convert_corrs_64 ( double _Complex * in_data, double _Complex * out_data )
{
    casacore::Vector<casacore::Complex> casa_data_in ( _in_format_len );
    casacore::Vector<casacore::Complex> casa_data_out ( _out_format_len );
    for ( std::size_t corr = 0; corr < _in_format_len; ++corr ) {
        casa_data_in[corr] = casacore::Complex ( creal ( in_data[corr] ),cimag ( in_data[corr] ) );
    }
    _conv->convert ( casa_data_out,casa_data_in );
    for ( std::size_t corr = 0; corr < _out_format_len; ++corr ) {
        out_data[corr] = static_cast<double> ( casa_data_out[corr].real() ) + static_cast<double> ( casa_data_out[corr].imag() ) * _Complex_I;
    }
}

void give_psf_vis_32 ( size_t out_format_len,
                       int * out_format,
                       float _Complex * out_data )
{
    casa::Vector<casa::Int> out ( out_format_len );
    casacore::Vector<casacore::Complex> casa_data_out ( out_format_len );
    for ( std::size_t corr = 0; corr < out_format_len; ++corr ) {
        out[corr] = static_cast<casa::Int> ( out_format[corr] );
    }
    casa::Vector<casa::Int> in ( 4 );
    in[0] = casacore::Stokes::I;
    in[1] = casacore::Stokes::Q;
    in[2] = casacore::Stokes::U;
    in[3] = casacore::Stokes::V;
    casa::Vector<casacore::Complex> casa_data_in ( 4 );
    casa_data_in[0] = 1 + 0 * _Complex_I;
    casa_data_in[1] = 0 + 0 * _Complex_I;
    casa_data_in[2] = 0 + 0 * _Complex_I;
    casa_data_in[3] = 0 + 0 * _Complex_I;
    casacore::StokesConverter conv ( out,in,true );
    conv.convert ( casa_data_out,casa_data_in );
    for ( std::size_t corr = 0; corr < out_format_len; ++corr ) {
        out_data[corr] = static_cast<float> ( casa_data_out[corr].real() ) + static_cast<float> ( casa_data_out[corr].imag() ) * _Complex_I;
    }
}

void give_psf_vis_64 ( size_t out_format_len,
                       int * out_format,
                       double _Complex * out_data )
{
    casa::Vector<casa::Int> out ( out_format_len );
    casacore::Vector<casacore::Complex> casa_data_out ( out_format_len );
    for ( std::size_t corr = 0; corr < out_format_len; ++corr ) {
        out[corr] = static_cast<casa::Int> ( out_format[corr] );
    }
    casa::Vector<casa::Int> in ( 4 );
    in[0] = casacore::Stokes::I;
    in[1] = casacore::Stokes::Q;
    in[2] = casacore::Stokes::U;
    in[3] = casacore::Stokes::V;
    casa::Vector<casacore::Complex> casa_data_in ( 4 );
    casa_data_in[0] = 1 + 0 * _Complex_I;
    casa_data_in[1] = 0 + 0 * _Complex_I;
    casa_data_in[2] = 0 + 0 * _Complex_I;
    casa_data_in[3] = 0 + 0 * _Complex_I;
    casacore::StokesConverter conv ( out,in,true );
    conv.convert ( casa_data_out,casa_data_in );
    for ( std::size_t corr = 0; corr < out_format_len; ++corr ) {
        out_data[corr] = static_cast<double> ( casa_data_out[corr].real() ) + static_cast<double> ( casa_data_out[corr].imag() ) * _Complex_I;
    }
}