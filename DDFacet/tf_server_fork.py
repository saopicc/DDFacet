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

import sys

try:
    import tensorflow as tf
    # Need to load the library containing the tensorflow RIME ops
    from montblanc.impl.rime.tensorflow import load_tf_lib
    load_tf_lib()
    server = tf.train.Server.create_local_server()
    # Print the server target to stdout
    # which will be read by the parent process
    print server.target
    sys.stdout.flush()
    server.join()
except:
    # Print any errors -- the parent process will complain
    # if nothing matching a server target is received
    print 'Error importing tensorflow :%s' % (sys.exc_info(),)