'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA-SA, Rhodes University

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
'''

import ctypes
import numpy as np

class Obj():
    def __init__(self,a):
        self.__array_interface__={
            "shape":(a.size,),
            "data":a.__array_interface__["data"],
            "typestr":a.__array_interface__["typestr"]
        }
        print self.__array_interface__

def test():
    a=np.random.randn(4,4)
    b=np.asarray(Obj(a))
    print a
    print b
    

class Obj2():
    def __init__(self,D):
        self.__array_interface__=D


#########################

import ctypes
 
class MutableString(object):
  def __init__(self, s):
    # Allocate string memory
    self._s = ctypes.create_string_buffer(s)
    print ctypes.addressof(self._s)
    self.__array_interface__ = {
      # Shape of the array
      'shape': (len(s),),

      # Address of data,
      # the memory is not read-only
      'data': (ctypes.addressof(self._s), False),
        
      # Stores 1-byte unsigned integers.
      # "|" indicates that Endianess is
      # irrelevant for this data-type.
      'typestr': '|u1',
      }
    print self.__array_interface__

  def __str__(self):
    "Convert to a string for printing."
    return str(buffer(self._s))


def testStr():
    m = MutableString('abcde')
    print m
    am = np.asarray(m)
    print am
    am += 2
 
    print am
    print m
