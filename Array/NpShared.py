import SharedArray
import ModColor

def ToShared(Name,A):

    try:
        a=SharedArray.create(Name,A.shape,dtype=A.dtype)
    except:
        print ModColor.Str("File %s exists, delete it..."%Name)
        DelArray(Name)
        a=SharedArray.create(Name,A.shape,dtype=A.dtype)


    a[:]=A[:]
    return a

def DelArray(Name):
    SharedArray.delete(Name)
    
def GiveArray(Name):
    return SharedArray.attach(Name)
