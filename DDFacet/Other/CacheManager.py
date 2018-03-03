'''
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
'''

import os, os.path, subprocess
import cPickle
import collections

from DDFacet.Other import MyLogger, ModColor
log = MyLogger.getLogger("CacheManager")


class CacheManager (object):
    """
    # CacheManager

    This is a simple disk-based cache management class. Usage is as follows:

        cache = CacheManager("/cache/directory")

    This creates a cache based on the specified path. Directory is created if it doesn't exist.
    If you create CacheManager with reset=True, the contents of the directory are deleted.

    Typically, you would use the cache in the following pattern:

        path, valid = cache.checkCache("foo", hashvalue)
        if valid:
            ## your data is on disk and good, load it from "path". E.g.
            data = numpy.load(path)
        else:
            ## your data must be recomputed. Recompute it and save it to "path".
            data = expensiveComputation()
            numpy.save(path, data)
            cache.saveCache("foo")  # mark the cache as valid

    So: checkCache() returns a path to the item (/cache/directory/foo, in this case), and a flag telling
    you if the saved item is valid. The item is considered valid if /cache/directory/foo exists, **and**
    /cache/directory/foo.hash exists, **and** contains the same value as the supplied "hashvalue". Thus,
    if hashvalue has changed w.r.t. the stored hash, then the cache is invalid.

    Hashvalue may be any Python object supporting the comparison operator (for example, a dict).
    Typically, this would be a dict of parameters any changes in which should cause the data item
    to be recomputed.

    The saveCache("foo") call is vitally important! The cache manager cannot know by itself when
    you've successfully written to the cache. By calling saveCache(), you're telling it that your data
    has been safely written to /cache/directory/foo. The cache manager then takes the hashvalue you
    supplied in the previous checkCache() call, and saves it to /cache/directory/foo.hash.


    # Using cache manager in DDFacet

    DDFacet caches the following data items:

        * W-kernels & other facet-related data

        * BDA mappings for gridding and degridding, flags, Jones matrices

    Each MS has a top-level cache named mspath.ddfcache, and per-chunk caches named
    mspath.ddfcache/Fx:Dy:n:m, where F is field ID, D is DDID, and n:m are row numbers.
    The per-chunk caches are used when iterating over MSs during gridding/degridding.

    Each ClassMS object has an ms.maincache attribute, and an ms.cache attribute.
    The former corresponds to that MS's top-level cache. The latter corresponds to
    the cache of the current chunk being iterated over.

    The ClassVisServer object has a VS.maincache an a VS.cache attribute. VS.maincache
    points to the maincache of the first MS in the list. This is used to cache facet data.
    VS.cache points to the cache of the current chunk of the current MS being iterated over.
    This is used to cache the various mappings.

    To form up hashvalues, the global dict (GD) object is very convenient. Sections of GD
    that influence a particular cache item are used as the hashvalue in each case.

    Running with DeleteDDFProducts=1 causes all caches to be reset.
    """

    def __init__(self, dirname, reset=False, cachedir=None, nfswarn=False):
        """
        Initializes cache manager.

        Args:
            dirname: directory in which caches are kept
            reset: if True, cache is reset upon first access
            cachedir: if set, caches things under cachedir/dirname. Useful for fast local storage.
            nfswarn: if True and directory is NFS mounted, prints a warning
        """
        # strip trailing slashes
        while dirname[-1] == "/":
            dirname = dirname[:-1]
        if cachedir:
            dirname = os.path.join(cachedir, os.path.basename(dirname))
        self.dirname = dirname
        self.hashes = {}
        self.pid = os.getpid()
        if not os.path.exists(dirname):
            print>>log, ("cache directory %s does not exist, creating" % dirname)
            os.mkdir(dirname)
        else:
            if reset:
                print>> log, ("clearing cache %s, since we were asked to reset the cache" % dirname)
                os.system("rm -fr "+dirname)
                os.mkdir(dirname)
        # check for NFS system and print warning
        if nfswarn:
            try:
                fstype = subprocess.check_output(("stat --file-system --format=%T " + dirname).split()).strip()
            except:
                print>> log, ModColor.Str("WARNING: unable to determine filesystem type for %s" % dirname, col="red",
                                          Bold=True)
                fstype = "unknown"
            if fstype == "nfs":
                print>> log, ModColor.Str("WARNING: cache directory %s is mounted via NFS." % dirname, col="red",
                                          Bold=True)
                print>> log, ModColor.Str("This may cause performance issues. Consider using the --Cache-Dir option.",
                                          col="red",
                                          Bold=True)

    @staticmethod
    def getElementName (name, **kw):
        """Helper function. Forms up a cache element filename as "NAME:KEY1_VALUE1:KEY2_VALUE2..."
        For example: getElementName("WTerm",facet=1)

        Args:
            name: name of cache element
            **kw: optional keywords, will be added to name as ":key_value"

        Returns:
            Concatenated Filename
        """
        return ":".join([name] + [ "%s_%s"%(key, value) for key, value in sorted(kw.items()) ])

    def getElementPath(self, name, **kw):
        """
        Forms up a full path for cache element 'name', with extra keywords. This is the element name plus
        the cache path. See getElementName() for usage.
        """
        return os.path.join(self.dirname, self.getElementName(name, **kw))

    def getCacheURL (self, name, **kw):
        """
        Forms up a URL for a disk-backed shared element. This takes the form of "file://PATH", where path is
        the cache element path as formed by getElementPath(). See the latter for usage.
        """
        return "file://" + self.getElementPath(name, **kw)

    def checkCache(self, name, hashkeys, directory=False, reset=False, ignore_key=False):
        """
        Checks if cached element named "name" is valid.

        Args:
            name: name of cache element
            hashkeys: dictionary of keys upon which the cached object depends. If a hash of the keys does not
                match the stored hash value, the cache is invalid and will be reset.
            directory: if True, cache is a directory and not a file. The directory will be created if it
                doesn't exist. If the cache is invalid, the contents of the directory will be deleted.
            reset: if True, cache item is deleted
            ignore_key: if True, keys are not compared, and cache is considered valid regardless.

        Returns:
            tuple of (path, valid)
            where path is a path to cache object (or cache directory)
            and valid is True if a valid cache exists
        """
        cachepath = self.getElementPath(name)
        hashpath = cachepath + ".hash"
        # convert hash keys into a single list
        hash = hashkeys
        self.hashes[name] = hashpath, hash
        # delete cache if explicitly asked to
        if reset:
            print>>log, "cache element %s will be explicitly reset" % cachepath
        else:
            if not os.path.exists(cachepath):
                print>>log, "cache element %s does not exist, will re-make" % cachepath
                if directory:
                    os.mkdir(cachepath)
                reset = True
            # check for stored hash
            if not reset:
                try:
                    storedhash = cPickle.load(file(hashpath))
                except:
                    print>>log, "cache hash %s invalid, will re-make" % hashpath
                    reset = True
            # check for hash match
            if not reset and not ignore_key and hash != storedhash:
                ListDiffer=[]
                for MainField, D1 in storedhash.iteritems():
                    if MainField not in hash:
                        ListDiffer.append("(%s: missing in hash)" % (str(MainField)))
                        continue
                    D0 = hash[MainField]
                    if type(D0) != type(D1):
                        ListDiffer.append("(%s: %s vs %s)" % (str(MainField), type(D0), type(D1)))
                    elif hasattr(D0,'iteritems'):
                        for key, value0 in D0.iteritems():
                            if key not in D1:
                                ListDiffer.append(
                                    "(%s.%s: %s vs missing)" % (str(MainField), str(key), str(D0[key])))
                            elif value0 != D1[key]:
                                ListDiffer.append("(%s.%s: %s vs %s)"%(str(MainField),str(key),str(value0),str(D1[key])))
                        for key in set(D1.keys()) - set(D0.keys()):
                            ListDiffer.append(
                                "(%s.%s: missing vs %s)" % (str(MainField), str(key), str(D1[key])))
                    else:
                        if D0 != D1:
                            ListDiffer.append("(%s: %s vs %s)"%(str(MainField),str(D0),str(D1)))
                for MainField in set(hash.keys()) - set(storedhash.keys()):
                    ListDiffer.append(
                        "(%s: missing in stored hash)" % (str(MainField)))

                print>>log, "cache hash %s does not match, will re-make" % hashpath
                print>>log, "  differences in parameters (Param: this vs cached): %s"%" & ".join(ListDiffer)
                
                reset = True
            # if resetting cache, then mark new hash value for saving (will be saved in flushCache),
            # and remove any existing cache/hash
        if reset:
            if os.path.exists(hashpath):
                os.unlink(hashpath)
            if os.path.exists(cachepath):
                if directory:
                    if os.system("rm -fr %s" % cachepath):
                        raise OSError,"Failed to remove cache directory %s. Check permissions/ownership." % cachepath
                    os.mkdir(cachepath)
                else:
                    os.unlink(cachepath)

        # store hash
        self.hashes[name] = hashpath, hash, reset
        return cachepath, not reset

    def saveCache(self, name=None):
        """
        Saves cache hash to disk. Meant to be called after a cache object has been successfully written to.

        Args:
            name: name of cache object. If None, all accumulated objects are flushed.

        Returns:

        """
        names = [name] if name else self.hashes.keys()
        for name in names:
            hashpath, hash, reset = self.hashes[name]
            if reset:
                cPickle.dump(hash, file(hashpath, "w"))
                print>>log, "writing cache hash %s" % hashpath
                del self.hashes[name]
