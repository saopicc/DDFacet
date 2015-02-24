import _pyGridder
import numpy as np

a=np.zeros((2,2))
x=np.array([0,0,0])
y=np.array([0,0,0])
w=np.ones_like(x)
print _pyGridder.pyGridderPoints(a.astype(np.float64),x,y,w.astype(np.float64),0)


S=12
Np=10
a=np.zeros((S,S))
x=np.int64(np.random.rand(Np)*S)
y=np.int64(np.random.rand(Np)*S)
w=np.ones_like(x)
print x,y
print _pyGridder.pyGridderPoints(a.astype(np.float64),x.astype(np.int32),y.astype(np.int32),w.astype(np.float64),2)

print b.min(),b.max()
print b
# import pylab
# pylab.imshow(b)
# pylab.draw()
# pylab.show()

