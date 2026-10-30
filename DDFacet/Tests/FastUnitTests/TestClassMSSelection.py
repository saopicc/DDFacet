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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import shutil
import tempfile

import numpy as np
from casacore.tables import default_ms

from DDFacet.Data import ClassMS


class _ListHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def testEmptySelectionNamesFieldsPresent():
    """An MS without rows for the selected field reports the fields it does have."""
    tmpdir = tempfile.mkdtemp()
    handler = _ListHandler()
    logger = logging.getLogger("app.ClassMS")
    logger.addHandler(handler)
    try:
        msname = os.path.join(tmpdir, "field12.ms")
        tab = default_ms(msname)
        tab.addrows(3)
        tab.putcol("FIELD_ID", np.array([1, 1, 2], np.int32))
        tab.putcol("DATA_DESC_ID", np.zeros(3, np.int32))
        tab.close()

        ms = ClassMS.ClassMS(msname, Col="DATA", DoReadData=False, Field=0, DDID=0)

        assert ms.empty
        skipped = [m for m in handler.messages if "no rows, skipping" in m]
        assert len(skipped) == 1
        assert "present in this MS: 1/0, 2/0" in skipped[0]
    finally:
        logger.removeHandler(handler)
        shutil.rmtree(tmpdir)
