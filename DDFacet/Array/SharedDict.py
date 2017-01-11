import os, os.path, cPickle

import NpShared
import numpy as np


SHM_PREFIX = "/dev/shm/"
SHM_PREFIX_LEN = len(SHM_PREFIX)

def _to_shm (path):
    """Helper function, converts /dev/shm/name to shm://name"""
##    return "shm://" + path[SHM_PREFIX_LEN:]
    # it seems that shm_open() does not support subdirectories. open("/dev/shm/a") should have
    # the same effect though (even if it is Linux-specific), so use that instead
    return "file://" + path

class SharedDict (dict):
    def __init__ (self, path, reset=True):
        dict.__init__(self)
        if path.startswith(SHM_PREFIX):
            self.path = path
        else:
            self.path = SHM_PREFIX + path
        if reset or not os.path.exists(self.path):
            self.clear()
        else:
            self.reload()

    def clear(self):
        if os.path.exists(self.path):
            os.system("rm -fr " + self.path)
        os.mkdir(self.path)

    def reload(self):
        """(Re)initializes dict with items from path"""
        dict.clear(self)
        # scan our subdirectory for items
        for name in os.listdir(self.path):
            filepath = os.path.join(self.path, name)
            # directory item -- is a nested SharedDict
            if os.path.isdir(filepath):
                dict.__setitem__(self, name, SharedDict(path=filepath, reset=False))
            # pickle item -- load directly
            elif name.endswith("._p"):
                dict.__setitem__(self, name[:-3], cPickle.load(file(filepath)))
            # array item -- attach as shared
            elif name.endswith("._a"):
                # strip off /dev/shm/ at beginning of path
                dict.__setitem__(self, name[:-3], NpShared.GiveArray(_to_shm(filepath)))
            else:
                print "Unknown shared dict entry "+filepath

    def __setitem__ (self, item, value):
        if type(item) is not str:
            raise TypeError,"SharedDict only supports string keys"
        dict.__setitem__ (self, item, value)
        filepath = os.path.join(self.path, item)
        # for arrays, copy to a shared array
        if isinstance(value, np.ndarray):
            NpShared.ToShared(_to_shm(filepath+"._a"), value)
        # for shared dicts, force use of addSubDict
        elif isinstance(value, SharedDict):
            raise TypeError,"shared sub-dicts must be initialized with addSubDict"
        # all other types, just use pickle
        else:
            cPickle.dump(value, file(filepath+"._p", "w"), 2)

    def addSubDict (self, name):
        if type(name) is not str:
            raise TypeError,"SharedDict only supports string keys"
        filepath = os.path.join(self.path, name)
        subdict = SharedDict(filepath, reset=True)
        dict.__setitem__(self, name, subdict)
        return subdict

    def addSharedArray (self, name, shape, dtype):
        if type(name) is not str:
            raise TypeError, "SharedDict only supports string keys"
        filepath = os.path.join(self.path, name)
        array = NpShared.CreateShared(_to_shm(filepath+"._a"), shape, dtype)
        dict.__setitem__(self, name, array)
        return array


def testSharedDict ():
    dic = SharedDict("foo")
    dic['a'] = 'a'
    dic['b'] = (1,2,3)
    dic['c'] = np.array([1,2,3,4])
    subdict = dic.addSubDict('subdict')
    subdict['a'] = 'aa'
    subdict['b'] = ('a', 1, 2, 3)
    subdict['c'] = np.array([1, 2, 3, 4, 5, 6])
    subdict2 = subdict.addSubDict('subdict2')
    subdict2['a'] = 'aaa'

    arr = subdict.addSharedArray("foo",(4, 4), np.float32)
    arr.fill(1)

    print dic

    other_view = SharedDict("foo", reset=False)
    print other_view

