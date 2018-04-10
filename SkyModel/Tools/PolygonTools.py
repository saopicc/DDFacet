import Polygon
import numpy as np

def test():
    
    RadiusTot=1.
    Poly=np.array([[-RadiusTot,-RadiusTot],
                   [+RadiusTot,-RadiusTot],
                   [+RadiusTot,+RadiusTot],
                   [-RadiusTot,+RadiusTot]])*1

    Line=np.array([[0.,0.],
                   [5.,5.]])

    print CutLineInside(Poly,Line)

def GiveABLin(P0,P1):
    x0,y0=P0
    x1,y1=P1
    a=(y1-y0)/(x1-x0)
    b=y1-a*x1
    return b,a

def GiveB(P0,P1):
    x0,y0=P0
    x1,y1=P1

    B=np.array([x0-x1,y0-y1])
    B/=np.sqrt(np.sum(B**2))
    if B[0]<0: B=-B
    return B

def CutLineInside(Poly,Line):
    P=Polygon.Polygon(Poly)

    dx=1e-4
    PLine=np.array(Line.tolist()+Line.tolist()[::-1]).reshape((4,2))
    #PLine[2,0]+=dx
    #PLine[3,0]+=2*dx
    PLine[2:,:]+=np.random.randn(2,2)*1e-6
    P0=Polygon.Polygon(PLine)
    PP=np.array(P0&P)[0].tolist()
    PP.append(PP[0])

    B0=GiveB(Line[0],Line[1])
    B=[GiveB(PP[i],PP[i+1]) for i in range(len(PP)-1)]

    PLine=[]
    for iB in range(len(B)):
        d=np.sum((B[iB]-B0)**2)
        print d,PP[iB],PP[iB+1]
        if d==0:
            PLine.append([PP[i],PP[i+1]])


    LOut=np.array(PLine[0])

    return LOut
    
