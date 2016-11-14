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