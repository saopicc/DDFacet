# various module entry points

def ddf_main():
    from DDFacet import DDF
    DDF.driver()

def cleanshm_main():
    from DDFacet import CleanSHM
    CleanSHM.driver()

def memmonitor_main():
    from DDFacet import MemMonitor
    MemMonitor.driver()

def restore_main():
    from DDFacet import Restore
    Restore.driver()