from __future__ import division, absolute_import, print_function
def reformat(ssin,slash=True,LastSlash=True):
    ss=ssin.split("/")
    ss=[a for a in ss if a != ""]
    ss="/".join(ss)+"/"
    if ssin[0]=="/": ss="/"+ss
    if not(slash):
        ss=ss[1:-1]
    if not(LastSlash):
        if ss[-1]=="/": ss=ss[0:-1]
    return ss
