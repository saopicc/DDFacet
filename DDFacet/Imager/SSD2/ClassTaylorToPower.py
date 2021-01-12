#from sympy import *
import sympy
from DDFacet.Other import logger
log=logger.getLogger("ClassInitSSDModel")
import copy

def test():
    CTP=ClassTaylorToPower(4)
    CTP.Compute()

class ClassTaylorToPower():
    def __init__(self,N):
        self.N=N

    def Compute(self):
        N=self.N
        Ss=" ".join(["a%i"%i for i in range(self.N)])
        St=" ".join(["b%i"%i for i in range(self.N)])
        S=Ss+" "+St+" nu nu0"
        Se=", ".join(S.split(" "))
        log.print("Declaring variables: %s"%Se)
        Sexec="%s = sympy.symbols('%s')"%(Se,S)
        exec(Sexec)
        #a0, a1, a2, nu, nu0 =sympy.symbols('a0 a1 a2 nu nu0')

        SPow="+".join(["a%i*sympy.Pow(sympy.log(nu/nu0),%i) "%(i,i-1) for i in range(1,N)])
        Ex="e=a0*sympy.Pow((nu/nu0),%s)"%SPow
        #Ex="e=a0*Pow((nu/nu0),2)"
        #exec("import sympy; %s"%Ex,locals())
        #exec("import sympy; %s"%Ex,locals())
        Ex="a0*sympy.Pow((nu/nu0),%s)"%SPow
        e=eval("%s"%Ex)
        log.print("Spectral expression (log): %s"%e)
        
        et=eval("e.series(x=nu, x0=nu0, n=N)")
        et0=sympy.Add(*(et.args[0:N]))

        P = eval("sympy.Poly(et0, nu)")
        pc0=P.coeffs()

        
        Ex="+".join(["b%i*sympy.Pow((nu-nu0)/nu0,%i) "%(i,i) for i in range(0,N)])
        e1=eval(Ex)
        log.print("Spectral expression (lin): %s"%e1)
        P1=eval("sympy.Poly(e1,nu)")
        pc1=P1.coeffs()
        L=[sympy.Eq(pc0[i], pc1[i]) for i in range(0,N)]
        La=",".join(["a%i"%i for i in range(N)])
        S=sympy.solve(L,La)[0]
        log.print("Solutions")
        SLa=" ".join(["a%i"%i for i in range(self.N)])
        La=eval("sympy.symbols('%s')"%(SLa))
        print(La)
        LLambda=[]
        for ik,k in enumerate(La):
            log.print("   %s = %s"%(k,S[k].simplify()))
            f=eval('sympy.lambdify(%s,S[k],"numpy")'%str(tuple(La)))
            LLambda.append(f)
            
        self.LLambda=LLambda

    def LinPolyCube2LogPolyCube(self,Cube):
        NTerms,npol,nx,ny=Cube.shape
        Lp=[]
        for iTerm in range(NTerms):
            Lp=Cube[iTerm,0,:,:]
        Lp=tuple(Lp)
        CubeOut=np.zeros_like(Cube)
        for iTerm in range(NTerms):
            CubeOut[iTerm,0,:,:]=self.LLambda[iTerm](*Lp)
        return CubeOut
    
# et.factor()
# et.simplify()
# solve([Eq(x + 5*y, 2), Eq(-3*x + 6*y, 15)], [x, y])
# et=e.series(x=nu, x0=nu0, n=3)
# et
# et.expand()
# et
# b0, b1, b2 = symbols('b0 b1 b2')
# et
# e0=b0+b1*(nu-nu0)/nu0+b2*((nu-nu0)/nu0)**2
# solve([Eq(e0, e)], [a0,a1,a2])
# e
# solve([Eq(e0, et)], [a0,a1,a2])
# et
# et.args
# et.args[0:3]
# "+".list(et.args[0:3])
# et.args[0:3]
# Add(et.args[0:3])
# Add(*(et.args[0:3]))
# et0=Add(*(et.args[0:3]))
# solve([Eq(e0, et0)], [a0,a1,a2])
# et0.as_poly()
# et0.as_poly().simplify()
# et0.as_poly()
# P0=et0.as_poly()
# P0.coeff()
# P0.coeff?
# P0.coeff(nu)
# P0.coeff(nu0)
# P0.coeffs()
# %hist
# solve([Eq(x + x**2, y)], [x, y])
# solve([Eq(e0, et0)], [b0,b1,b2])
# e0
# et0
# et0.simplify()
# solve([Eq(e0, et0)], [a0,a1,a2])
# a, x,y,b = symbols('a x y b')
# solve([Eq(, et0)], [b0,b1,b2])
# solve([Eq(a1*x+a0, b0+b1x)], [a0,a1])
# solve([Eq(a1*x+a0, b0+b1*x)], [a0,a1])
# solve([Eq(a0+a1*x, b0+b1*x)], [a0,a1])
# solve([Eq(a0+a1*x - b0+b1*x, 0)], [a0,a1])
# solve([Eq(a0+a1*x - b0+b1*x, 0)], [a0,a1,x])
# P0.
# P0.coeffs()
# P0.coeff_monomial()
# P0.coeff?
# P0.coeff??
# et0.collect((nu-nu0)/nu0)
# et0
# et0.collect((nu-nu0)/nu0).coeff((nu-nu0)/nu0,0)
# et0.collect((nu-nu0)/nu0).coeff((nu-nu0)/nu0,1)
# et0.collect((nu-nu0)/nu0).coeff((nu-nu0)/nu0,2)
# et0.collect().coeff((nu-nu0)/nu0,2)
# et0.collect(nu).coeff((nu-nu0)/nu0,2)
# et0.collect(nu).coeff((nu-nu0)/nu0,1)
# et0.collect(nu).coeff((nu-nu0)/nu0,2)
# et0.collect((nu-nu0)/nu0).coeff((nu-nu0)/nu0,0)
# et0
# et0.collect(((nu-nu0)/nu0)**2).coeff((nu-nu0)/nu0,0)
# et0.collect(((nu-nu0)/nu0)**2).coeff((nu-nu0)/nu0,1)
# et0.collect(((nu-nu0)/nu0)**2).coeff((nu-nu0)/nu0,2)
# et
# et0=Add(*(et.args[0:3]))
# et0
# et0.collect(((nu-nu0)/nu0)**2)
# et0.collect(((nu-nu0)/nu0))
# et0.collect(1)
# et0.factor()
# et0.factor?
# et0.factor?.
# et0.factor??
# et0.collect(x)
# et0.collect(nu)
# et0.collect(nu,1)
# et0.collect??
# solve([Eq(a0+a1*x, b0+b1*x)], [a0,a1,b0,b1])
# solve([Eq(a0+a1*x, b0+b1*x)], [a0])
# solve([Eq(a0+a1*x, b0+b1*x)], [a0,b0])
# solve([Eq(a0, b0),Eq(a1, b1)], [a0,a1])
# et0
# P = Poly(et0, (nu - nu0)/nu0)
# P.coeffs()
# P = Poly(et0, nu)
# P.coeffs()
# X=(nu - nu0)/nu0
# P = Poly(et0, X)
# P.coeffs()
# P1=b0+b1*(nu-nu0)/nu0+b2*((nu-nu0)/nu0)**2
# P1=Poly(b0+b1*(nu-nu0)/nu0+b2*((nu-nu0)/nu0)**2)
# P1.coeffs()
# P1=Poly(b0+b1*(nu-nu0)/nu0+b2*((nu-nu0)/nu0)**2,nu)
# P1.coeffs()
# pc0=P.coeffs()
# pc1=P1.coeffs()
# solve([Eq(pc0[0], pc1[0]), Eq(pc0[1], pc1[1]), Eq(pc0[2], pc1[2])],[a0,a1,a2])
# print([Eq(pc0[0], pc1[0]), Eq(pc0[1], pc1[1]), Eq(pc0[2], pc1[2])],[a0,a1,a2])
# print([Eq(pc0[0], pc1[0]), Eq(pc0[1], pc1[1]), Eq(pc0[2], pc1[2])])
# print(pc0[0], pc1[0])
# print(pc0[1], pc1[1])
# pc1
# type(pc1)
# len(pc1)
# pc1[1]
# pc0[1]
# type(pc0)
# len(pc0)
# pc0
# pc0=P.coeffs()
# P.coeffs()
# P = Poly(et0, nu)
# pc0=P.coeffs()
# pc0
