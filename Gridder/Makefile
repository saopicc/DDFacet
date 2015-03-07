# ---- Link --------------------------- 
_pyGridder.so:  Gridder.o 
	gcc -shared Gridder.o -o _pyGridder.so

# ---- gcc C compile ------------------
Gridder.o:  Gridder.c Gridder.h
	gcc -O3 -msse -msse2 -msse2avx -msse3 -msse4.2 -mssse3 -march=corei7 -mtune=corei7 -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC	

#gcc -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC

# # ---- Link --------------------------- 
# _pyGridder1D.so:  Gridder1D.o 
# 	gcc -shared Gridder1D.o -o _pyGridder1D.so

# # ---- gcc C compile ------------------
# Gridder1D.o:  Gridder1D.c Gridder1D.h
# 	gcc -c Gridder1D.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC

