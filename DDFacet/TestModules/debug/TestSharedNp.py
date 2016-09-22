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
