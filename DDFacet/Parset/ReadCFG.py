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
    Names of built-in functions are _not_ interpreted as functions!
    """
    try:
        value = eval(string, {}, {})
        if type(value) is type(all):  # do not interpret built-in function names
            return string
        return value
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

    # interpret explicit types
    if type:
        # make sure None string is still None
        if type is str and string == "None" or string == "none":
            return None, attrs
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
        self.value_dict = self.DicoPars = OrderedDict()   # call it DicoPars for compatibility with old testing code
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
        self.Config = config = ConfigParser.ConfigParser(dict_type=OrderedDict)
        config.optionxform = str
        success = config.read(self.filename)
        self.success = bool(len(success))
        if self.success:
            self.sections = config.sections()
            for section in self.sections:
                self.value_dict[section], self.attr_dict[section] = self.read_section(config, section)
        # now migrate from previous versions
        self.version = self.value_dict.get('Misc', {}).get('ParsetVersion', 0.0)
        if self.version != 0.2:
            if self.version != 0.1:
              self._migrate_ancient_0_1()
            self._migrate_0_1_0_2()
            self.migrated = self.version
            self.version = self.value_dict['Misc']['ParsetVersion'] = 0.2
        else:
            self.migrated = None
        # if "Mode" not in self.value_dict["Output"] and "Mode" in self.value_dict["Image"]:
        #     self.value_dict["Output"]["Mode"] = self.value_dict["Image"]["Mode"]
        #     del self.value_dict["Image"]["Mode"]

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

    def set (self, section, option, value):
        self.value_dict.setdefault(section,{})[option] = value

    def write (self, f):
        """Writes the Parset out to a file object"""
        for section, content in self.value_dict.iteritems():
            f.write('[%s]\n'%section)
            for option, value in content.iteritems():
                attrs = self.attr_dict.get(section, {}).get(option, {})
                if option[0] != "_" and not attrs.get('cmdline_only') and not attrs.get('alias_of'):
                    f.write('%s = %s \n'%(option, str(value)))
            f.write('\n')

    def _makeSection (self, section):
        """
        Helper method for migration: makes a new section
        """
        for dd in self.value_dict, self.attr_dict:
            dd.setdefault(section, OrderedDict())
        return section

    def _renameSection (self, oldname, newname):
        """
        Helper method for migration: renames a section. If the new section already exists, merges options into
        it.
        """
        for dd in self.value_dict, self.attr_dict:
            if oldname in dd:
                dd.setdefault(newname, OrderedDict()).update(dd.pop(oldname))
        return newname

    def _del (self, section, option):
        """
        Helper method for migration: removes an option
        """
        for dd in self.value_dict, self.attr_dict:
            if section in dd and option in dd[section]:
                dd[section].pop(option)

    def _rename (self, section, oldname, newname):
        """
        Helper method for migration: renames an option within a section. Optionally remaps option values using
        the supplied dict.
        """
        for dd in self.value_dict, self.attr_dict:
            if section in dd and oldname in dd[section]:
                dd[section][newname] = dd[section].pop(oldname)

    def _remap (self, section, option, remap):
        """
        Helper method for migration: remaps the values of an option
        """
        if section in self.value_dict and option in self.value_dict[section]:
            value = self.value_dict[section][option]
            if value in remap:
                self.value_dict[section][option] = remap[value]

    def _move (self, oldsection, oldname, newsection, newname):
        """
        Helper method for migration: moves an option to a different section
        """
        for dd in self.value_dict, self.attr_dict:
            if oldsection in dd and oldname in dd[oldsection]:
                dd.setdefault(newsection, OrderedDict())[newname] = dd[oldsection].pop(oldname)
                
    def _migrate_0_1_0_2 (self):
        self._move("Image", "Mode", "Output", "Mode")

    def _migrate_ancient_0_1 (self):
        """
        Migrates contents from "old-style" (pre-issue255) parset to 0.1 parset
        """

        self._makeSection("Misc")
        self._makeSection("Hogbom")
        self._makeSection("Facets")
        self._makeSection("Weight")
        self._makeSection("RIME")
        self._makeSection("Predict")

        section = "Parallel"
        self._del(section, "Enable")  # deprecated. Use NCPU=1 instead

        section = self._renameSection("Caching", "Cache")
        self._rename(section, "ResetCache", "Reset")
        self._rename(section, "CachePSF", "PSF")
        self._rename(section, "CacheDirty", "Dirty")
        self._rename(section, "CacheVisData", "VisData")

        section = self._renameSection("VisData", "Data")
        self._rename(section, "MSName", "MS")
        self._del(section, "MSListFile")  # deprecated. Use MS=list.txt instead
        self._move(section, "WeightCol", "Weight", "ColName")
        self._move(section, "PredictColName", "Predict", "ColName")

        section = self._renameSection("DataSelection", "Selection")

        section = self._renameSection("Images", "Output")
        self._move(section, "AllowColumnOverwrite", "Predict", "Overwrite")
        self._move(section, "PredictModelName", "Predict", "FromImage")
        self._rename(section, "ImageName", "Name")
        self._del(section, "SaveIms")  # deprecated
        self._rename(section, "SaveOnly", "Images")
        self._rename(section, "SaveImages", "Also")
        self._rename(section, "SaveCubes", "Cubes")
        self._del(section, "OpenImages")   # deprecated, do we really need this? Or make consistent with --Images-Save notation at least
        self._del(section, "DefaultImageViewer") # deprecated, do we really need this?
        self._del(section, "MultiFreqMap")  # deprecated

        section = self._renameSection("ImagerMainFacet", "Image")
        self._rename(section, "Npix", "NPix")

        section = self._renameSection("ImagerGlobal", "Image")
        self._move(section, "NFacets", "Facets", "NFacets")
        self._move(section, "Weighting", "Weight", "Mode")
        self._move(section, "Robust", "Weight", "Robust")
        self._move(section, "Super", "Weight", "SuperUniform")
        self._move(section, "MFSWeighting", "Weight", "MFS")
        self._move(section, "RandomSeed", "Misc", "RandomSeed")
        for x in "Precision", "PolMode":
            self._move(section, x, "RIME", x)
        self._move(section, "PredictMode", "RIME", "ForwardMode" )
        self._remap("RIME", "ForwardMode", {'DeGridder': 'BDA-degrid'})

        for x in "PSFOversize", "PSFFacets", "Padding", "Circumcision":
            self._move(section, x, "Facets", x)
        self._move(section, "DiamMaxFacet", "Facets", "DiamMax" )
        self._move(section, "DiamMinFacet", "Facets", "DiamMin" )

        self._move("DDESolutions", "DecorrMode", "RIME", "DecorrMode")

        section = self._renameSection("ImagerCF", "CF")

        section = self._renameSection("ImagerDeconv", "Deconv")
        self._rename(section, "MinorCycleMode", "Mode")
        self._remap(section, "Mode", {'MSMF': 'HMP'})
        self._rename(section, "SearchMaxAbs", "AllowNegative")
        self._move(section, "SidelobeSearchWindow", "Image", "SidelobeSearchWindow")

        section = self._renameSection("Compression", "Comp")
        self._del(section, "CompGridMode")  # deprecate for now, since only the BDA gridder works
        self._del(section, "CompDeGridMode")  # deprecate for now, since only the BDA degridder works
        self._rename(section, "CompGridDecorr", "GridDecorr")
        self._rename(section, "CompGridFOV", "GridFov")
        self._rename(section, "CompDeGridDecorr", "DegridDecorr")
        self._rename(section, "CompDeGridFOV", "DegridFOV")

        section = self._renameSection("MultiScale", "HMP")
        self._del(section, "MSEnable")  # deprecated. --Deconvolution-MinorCycle selects algorithm instead.
        self._move(section, "PSFBox", "Deconv", "PSFBox")
        # Alpha added

        section = self._renameSection("MultiFreqs", "Freq")  # options related to basic multifrequency imaging
        self._rename(section, "GridBandMHz", "BandMHz")
        self._rename(section, "NFreqBands", "NBand")
        self._rename(section, "NChanDegridPerMS", "NDegridBand")
        self._move(section, "Alpha", "HMP", "Alpha")
        self._move(section, "PolyFitOrder", "Hogbom", "PolyFitOrder")

        section = "Beam"
        self._rename(section, "BeamModel", "Model")
        self._rename(section, "NChanBeamPerMS", "NBand")

        section = self._renameSection("Logging", "Log")
        self._rename(section, "MemoryLogging", "Memory")
        self._rename(section, "AppendLogFile", "Append")

        section = self._renameSection("Debugging", "Debug")

