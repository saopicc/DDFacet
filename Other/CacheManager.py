import os
import os.path
import cPickle

from DDFacet.Other import MyLogger
log = MyLogger.getLogger("CacheManager")


class CacheManager (object):
    def __init__(self, dirname, reset=False):
        """
        Initializes cache manager.

        Args:
            dirname: directory in which caches are kept
            reset: if True, cache is reset upon first access
        """
        self.dirname = dirname
        self.hashes = {}
        if not os.path.exists(dirname):
            print>>log,("cache directory %s does not exist, creating"%dirname)
            os.mkdir(dirname)
        else:
            if reset:
                print>> log, ("clearing cache %s, since we were asked to reset the cache" % dirname)
                os.system("rm -fr "+dirname)
                os.mkdir(dirname)

    def checkCache (self, name, hashkeys, directory=False):
        """
        Checks if cached object named "name" is valid.

        Args:
            name: name of cached object
            hashkeys: dictionary of keys upon which the cached object depends. If a hash of the keys does not
                match the stored hash value, the cache is invalid and will be reset.
            dir: if True, cache is a directory and not a file. If cache needs to be reset, the contents
                of the dir will be deleted

        Returns:
            tuple of (path, valid)
            where path is a path to cache object (or cache directory)
            and valid is True if a valid cache exists
        """
        cachepath = os.path.join(self.dirname, name)
        hashpath = cachepath + ".hash"
        # convert hash keys into a single list
        hash = sorted(hashkeys.iteritems())
        reset = False
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
        if not reset and hash != storedhash:
            print>>log, "cache hash %s does not match, will re-make" % hashpath
            reset = True
        # if resetting cache, then mark new hash value for saving (will be saved in flushCache),
        # and remove any existing cache/hash
        if reset:
            if os.path.exists(hashpath):
                os.unlink(hashpath)
            if directory:
                if os.path.exists(cachepath):
                    os.system("rm -fr %s"%cachepath)
                os.mkdir(cachepath)
            self.hashes[name] = hashpath, hash
        return cachepath, not reset


    def saveCache (self, name=None):
        """
        Saves cache hash to disk. Meant to be called after a cache object has been successfully written to.

        Args:
            name: name of cache object. If None, all accumulated objects are flushed.

        Returns:

        """
        names = [name] if name else  self.hashes.keys()
        for name in names:
            hashpath, hash = self.hashes[name]
            cPickle.dump(hash, file(hashpath,"w"))
            print>>log,"writing cache hash %s" % hashpath
            del self.hashes[name]
