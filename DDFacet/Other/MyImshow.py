import pylab

def GiveVal(A,xin,yin):
    x,y=round(xin),round(yin)
    s=A.shape[0]-1
    cond=(x<0)|(x>s)|(y<0)|(y>s)
    if cond:
        value="out"
    else:
        value="%8.2f mJy"%(A.T[x,y]*1000.)
    return "x=%4i, y=%4i, value=%10s"%(x,y,value)

def imshow(ax,A,*args,**kwargs):
    ax.format_coord = lambda x,y : GiveVal(A,x,y)
    ax.imshow(A,*args,**kwargs)
