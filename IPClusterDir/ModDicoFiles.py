

def DictToFile(Dict,fout):
     f=open(fout,"w")
     #ll=sorted(Dict.iteritems(), key=lambda x: x[1]['id'])
     #Lkeys=[ll[i][0] for i in range(len(ll))]

     Lkeys=Dict.keys()
            
     for key in Lkeys:
         keyw=key
         f.write("%s = %s\n"%(keyw,Dict[key]))
     f.close()


def FileToDict(fname):
    f=file(fname,"r")
    Dict={}
    ListOut=f.readlines()
    order=[]
    i=0
    for line in ListOut:
        if line=='\n': continue
        key,val=line[0:-1].split("=")
        key=key.replace(" ","")
        key=key.replace(".","_")
        Dict[key]=int(val)#{"id":i,"show":1, "col":"", "help": "", "val":val}
        i+=1
    return Dict

