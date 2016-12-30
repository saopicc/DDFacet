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

import glob
import subprocess
# import DDFacet.DDFacet.cbuild
# 20 / 12 / 2016: No unit tests since gridder was removed, so I'm disabling this one
#def testRunCatchTests():
#  ls = glob.glob(DDFacet.cbuild.__path__[0] + "/Gridder/UnitTests/Test*")
#  for tester in ls:
#    ret = subprocess.call(tester,shell=True)
#    assert not ret, "C++ test unit %s failed" % tester