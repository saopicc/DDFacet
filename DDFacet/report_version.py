import os
import subprocess

from DDFacet import __version__


def report_version():
    # perhaps we are in a github with tags; in that case return describe
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        # work round possible unavailability of git -C
        result = subprocess.check_output('cd %s; git describe --tags' % path, shell=True, stderr=subprocess.STDOUT).rstrip()
    except subprocess.CalledProcessError:
        result = None
    if result is not None and 'fatal' not in result:
        # will succeed if tags exist
        return result
    else:
        # perhaps we are in a github without tags? Cook something up if so
        try:
            result = subprocess.check_output('cd %s; git rev-parse --short HEAD' % path, shell=True, stderr=subprocess.STDOUT).rstrip()
        except subprocess.CalledProcessError:
            result = None
        if result is not None and 'fatal' not in result:
            return __version__+'-'+result
        else:
            # we are probably in an installed version
            return __version__
