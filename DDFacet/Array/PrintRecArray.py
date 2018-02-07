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

def remove_field_name(a, name):
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b

from prettytable import PrettyTable
def Print(CatIn,RemoveFieldName='ChanFreq',HideList=None):
    if RemoveFieldName in CatIn.dtype.names:
        Cat=remove_field_name(CatIn, RemoveFieldName)
    else:
        Cat=CatIn

    Cat=Cat.copy()
    if HideList is not None:
        for field in HideList:
            Cat=remove_field_name(Cat, field)

    x = PrettyTable(Cat.dtype.names)
    for row in Cat: x.add_row(row)
    print x
