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

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
bold='\033[1m'
nobold='\033[0m'
Separator="================================%s=================================="
silent=0

def disableColors():
    global silent
    silent = 1
    
def Str(strin0,col="red",Bold=True):
    if silent==1: return strin0
    strin=str(strin0)
    if col=="red":
        ss=FAIL
    if col=="green":
        ss=OKGREEN
    elif col=="yellow":
        ss=WARNING
    elif col=="blue":
        ss=OKBLUE
    elif col=="green":
        ss=OKGREEN
    elif col=="white":
        ss=""

    ss="%s%s%s"%(ss,strin,ENDC)
    if Bold: ss="%s%s%s"%(bold,ss,nobold)
    return ss

def Sep(strin=None,D=1):
    if D!=1:
        return Str(Separator%("="*len(strin)))
    else:
        return Str(Separator%(strin))

def Title(strin,Big=False):
    print
    print
    if Big: print Sep(strin,D=0)
    print Sep(strin)
    if Big: print Sep(strin,D=0)
    print

def disable():
    HEADER = ''
    OKBLUE = ''
    OKGREEN = ''
    WARNING = ''
    FAIL = ''
    ENDC = ''
