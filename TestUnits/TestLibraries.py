import glob
import subprocess
import DDFacet.cbuild

def testRunCatchTests():
  ls = glob.glob(DDFacet.cbuild.__path__[0] + "/Gridder/TestUnits/Test*")
  for tester in ls:
    ret = subprocess.call(tester,shell=True)
    assert not ret, "C++ test unit %s failed" % tester