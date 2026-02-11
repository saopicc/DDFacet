# various module entry points

def ddf_main():
    from DDFacet import DDF
    DDF.driver()

def cleanshm_main():
    from DDFacet import CleanSHM
    CleanSHM.driver()

def iomonitor_main():
    from DDFacet import IOMonitor
    IOMonitor.driver()

def memmonitor_main():
    from DDFacet import MemMonitor
    MemMonitor.driver()

def restore_main():
    from DDFacet import Restore
    Restore.driver()
