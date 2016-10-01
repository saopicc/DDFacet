#!/usr/bin/env python
import optparse
import sys
import pickle


def read_options():
    desc = """DDFacet """

    opt = optparse.OptionParser(
        usage='Usage: %prog --Parset=somename.MS <options>',
        version='%prog version 1.0', description=desc)
    group = optparse.OptionGroup(
        opt,
        "* Data-related options",
     "Won't work if not specified.")
    group.add_option('--Parset', help='Input Parset [no default]', default='')
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option(
        '--PointingID',
        help='PointingID in case multiple pointing dataset [no default]',
     default=0)
    group.add_option(
        '--pngBaseDir',
        help='PNG directory [no default]',
     default='png')
    group.add_option('--EnablePlot', help='Enable matplotlib', default=0)
    # group.add_option('--TimeCode',help='TimeCode',default=[0,-1,1])
    group.add_option('--Incr', help='Increment in time-steps', default=10)
    opt.add_option_group(group)

    options, arguments = opt.parse_args()
    f = open("last_param.obj", "wb")
    pickle.dump(options, f)
    return options


def main(options=None):

    if options is None:
        f = open("last_param.obj", 'rb')
        options = pickle.load(f)

    if options.EnablePlot == 0:
        import matplotlib
        matplotlib.use('agg')

    TCode = (0, -1, int(options.Incr))

    import ClassMovieMachine
    MM = ClassMovieMachine.MovieMachine(
        ParsetFile=options.Parset,
        PointingID=options.PointingID,
        pngBaseDir=options.pngBaseDir,
     TimeCode=TCode)
    MM.MainLoop()


if __name__ == "__main__":
    options = read_options()
    main(options)
