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

import ConfigParser
from collections import OrderedDict
import re

def test():
    P=Parset()

### dead code?
# def FormatDico (DicoIn):
#     Dico=OrderedDict()
#     for key in DicoIn.keys():
#         Dico[key] = ParseConfigString(DicoIn[key])
#     return Dico


def parse_as_python(string, words_are_strings=False):
    """Tries to interpret string as a Python object. Returns value, or string itself if unsuccessful.
    """
    try:
        return eval(string, {}, {})
    except:
        return string

def parse_config_string(string, name='config', extended=True, type=None):
    """
    Parses configuration string, converting it to a Python object (i.e. bool, int, float, etc.).
    Can also create a dict of attributes formed as as described below.

    Config string can contain an optional embedded comment, preceded by a '#' symbol. This
    is converted into a docstring, and returned as attrs['doc']. Docstrings can contain optional
    #ATTRIBUTE:VALUE entries. These are extracted and removed from the docstring, and returned
    in the attrs dict.

    The 'type' attribute has special meaning. If present, it forces the string to be parsed
    as a specific type. This overrides the 'type' argument.

    If extended=False, then it DOESN'T attempt to parse out the docstring and attributes,
    but simply tries to interpret the object as the given type. This mode is suitable to
    parsing command-line arguments.

    Args:
        string:     string value to parse
        name:       name of option (used for error messages)
        extended:   if True, enables parsing of docstring and attributes
        type:       forces string to be interpreted as a specific type

    Returns:
        tuple of value, attribute_dict
    """
    if string is None:
        return None, {}

    attrs = {}
    if extended:
        # parse out docstring
        if "#" in string:
            string, docstring = string.split("#", 1)
            string = string.strip()
            # parse out attributes
            while True:
                docstring = docstring.strip()
                # find instance of #attr:value in docstring
                match = re.match("(.*)#(\w+):([^\s]*)(.*)", docstring, re.DOTALL)
                if not match:
                    break
                # extract attribute
                attrname, value = match.group(2), match.group(3)
                # #options:value, value is always treated as a string. Otherwise, treat as Python expression
                if attrname not in ("options",):
                    value = parse_as_python(value)
                attrs[attrname] = value
                # remove it from docstring
                docstring = match.group(1) + match.group(4)
        else:
            docstring = ""
        attrs["doc"] = docstring

        # if attributes contain a type, parse this out
        if 'type' in attrs:
            if attrs['type'] == 'string':
                attrs['type'] = str
            # type had better be a callable type object
            type = parse_as_python(attrs['type'])
            if not callable(type):
                raise ValueError("%s: invalid '#type:%s' attribute"%(name, attrs['type']))

    # if attributes contain an option list, enforce this
    if 'options' in attrs:
        opts = attrs['options'].split("|")
        if string not in opts:
            raise ValueError("%s: value %s not in options list"%(name, string))

    # make sure _Help is interpreted as a string
    if name == "_Help":
        return string, attrs

    if type:
        # make sure False/True etc. are interpreted as booleans
        if type is bool:
            return bool(parse_as_python(string)), attrs
        return type(string), attrs

    # Now, some kludges for backward compatibility
    # A,B,C and [A,B,C] are parsed to a list
    as_list = len(string)>1 and string[0] == '[' and string[-1] == ']'
    if as_list:
        string = string[1:-1]
    if as_list or "," in string:
        return [ parse_as_python(x) for x in string.split(",") ], attrs

    # Otherwise just interpret the value as a Python object if possible
    return parse_as_python(string), attrs


class Parset():
    def __init__(self, filename=None):
        """Creates parset, reads from file if specified"""
        self.value_dict = OrderedDict()
        self.attr_dict = OrderedDict()
        if filename:
            self.read(filename)

    def update_values (self, other, newval=True):
        """Updates this Parset with keys found in other parset. NB: does not update keys that are in other but
        not self."""
        for secname in self.value_dict.keys():
            for name, value in other.value_dict.get(secname, {}).iteritems():
                if name in self.value_dict[secname]:
                    attrs = self.attr_dict[secname].get(name,{})
                    if not attrs.get('cmdline_only'):
                        self.value_dict[secname][name] = value
                        # make sure aliases get copied under both names
                        alias = attrs.get('alias') or attrs.get('alias_of')
                        if alias:
                            self.value_dict[secname][alias] = value

    def read (self, filename):
        self.filename = filename
        config = ConfigParser.ConfigParser(dict_type=OrderedDict)
        config.optionxform = str
        success = config.read(self.filename)
        self.success = bool(len(success))
        if self.success:
            self.sections = config.sections()
            for section in self.sections:
                self.value_dict[section], self.attr_dict[section] = self.read_section(config, section)

    def read_section(self, config, section):
        """Returns two dicts corresponding to the given section: a dict of option:value,
        and a dict of option:attribute_dict"""
        dict_values = OrderedDict()
        dict_attrs = OrderedDict()
        for option in config.options(section):
            strval = config.get(section, option)
            # option names with an "|" in them specify a longhand alias
            if "|" in option:
                option, alias = option.split("|",1)
            else:
                alias = None
            dict_values[option], dict_attrs[option] = parse_config_string(strval, name=option)
            # if option has an alias, mke copy of both in dicts
            if alias:
                dict_attrs[option]['alias'] = alias
                dict_values[alias] = dict_values[option]
                dict_attrs[alias] = { 'alias_of': option }
        return dict_values, dict_attrs

    def _migrate_from_pre255 (self):
        """
        Migrates contents from "old-style" parset prior to issue #255 being resolved.
        """

        makeSection("Misc")

        section = "Parallel"
        delete(section, "Enable")  # deprecated. Use NCPU=1 instead

        section = renameSection("Caching", "Cache")
        rename(section, "ResetCache", "Reset")
        rename(section, "CachePSF", "PSF")
        rename(section, "CacheDirty", "Dirty")
        rename(section, "CacheVisData", "VisData")

        section = renameSection("VisData", "Data")
        rename(section, "MSName", "MS")
        delete(section, "MSListFile")  # deprecated. Use MS=list.txt instead
        # PredictFrom # migrated from --Images-PredictModelName

        section = renameSection("DataSelection", "Selection")

        section = renameSection("Images", "Output")
        move(section, "AllowColumnOverwrite", "Data", "Overwrite")
        move(section, "PredictModelName", "Data", "PredictFrom")
        rename(section, "ImageName", "Name")
        delete(section, "SaveIms")  # deprecated
        rename(section, "SaveOnly", "Images")
        rename(section, "SaveImages", "Also")
        rename(section, "SaveCubes", "Cubes")
        delete(section, "OpenImages")   # deprecated, do we really need this? Or make consistent with --Images-Save notation at least
        delete(section, "DefaultImageViewer") # deprecated, do we really need this?
        delete(section, "MultiFreqMap")  # deprecated

        section = renameSection("ImagerGlobal", "Image")
        rename(section, "Super", "SuperUniform")
        move(section, "RandomSeed", "Misc", "RandomSeed")

        section = renameSection("ImagerMainFacet", "Image")
        rename(section, "Npix", "NPix")

        section = renameSection("Compression", "Comp")
        delete(section, "CompGridMode")  # deprecate for now, since only the BDA gridder works
        delete(section, "CompDegridMode")  # deprecate for now, since only the BDA degridder works
        rename(section, "CompGridDecorr", "GridDecorr")
        rename(section, "CompGridFOV", "GridFov")
        rename(section, "CompDeGridDecorr", "DegridDecorr")
        rename(section, "CompDeGridFOV", "DegridFOV")

        section = renameSection("MultiFreqs", "Freq")  # options related to basic multifrequency imaging
        rename(section, "GridBandMHz", "BandMHz")
        rename(section, "NFreqBands", "NBand")
        rename(section, "NChanDegridPerMS", "NDegridBand")
        move(section, "Alpha", "HMP", "Alpha")
        move(section, "PolyFitOrder", "Hogbom", "PolyFitOrder")

        section = "Beam"
        rename(section, "BeamModel", "Model")
        rename(section, "NChanBeamPerMS", "NBand")

        section = renameSection("ImagerDeconv", "Deconv")
        rename(section, "MinorCycleMode", "Mode")
        rename(section, "SearchMaxAbs", "AllowNegative")
        move(section, "SidelobeSearchWindow", "Image", "SidelobeSearchWindow")

        section = renameSection("MultiScale", "HMP")
        delete(section, "MSEnable")  # deprecated. --Deconvolution-MinorCycle selects algorithm instead.
        move(section, "PSFBox", "Deconv", "PSFBox")
        # Alpha added

        section = "Hogbom"
        # PolyFitOrder added

        section = renameSection("Logging", "Log")
        rename(section, "MemoryLogging", "Memory")
        rename(section, "AppendLogFile", "Append")


        section = renameSection("Debugging", "Debug")

