import sys

pdb_advise = "Unexpected error. Dropping you into pdb for a post-mortem."

class UserInputError(Exception):
    pass

def _exc_handler(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type is SyntaxError or type is UserInputError:
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        print pdb_advise
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"


def enable_pdb_on_error(advise=None):
    if advise is not None:
        global pdb_advise
        pdb_advise = advise
    sys.excepthook = _exc_handler

def disable_pdb_on_error(advise=None):
    sys.excepthook = sys.__excepthook__

def is_pdb_enabled():
    return sys.excepthook == _exc_handler