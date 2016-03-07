all: _pyGridder.so _pyGridderSmear.so _pyGridderSmearPols.so _pyArrays.so WeightingCore.so

CFLAGS = -ggdb -O3

# ---- Link --------------------------- 
_pyGridder.so:  Gridder.o 
	gcc $(CFLAGS) -fopenmp -pthread -shared -lrt Gridder.o -o _pyGridder.so

# ---- gcc C compile ------------------
Gridder.o:  Gridder.c Gridder.h
#	gcc $(CFLAGS) -O3 -msse -msse2 -msse2avx -msse3 -msse4.2 -mssse3 -march=corei7 -mtune=corei7 -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC	

	gcc $(CFLAGS)  -lgomp -fopenmp -pthread -O3 -msse -msse2 -msse3 -msse4.2 -mssse3 -march=corei7 -mtune=corei7 --fast-math -lrt  -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC
#	gcc $(CFLAGS)  -lgomp -fopenmp -pthread -Ofast --fast-math  -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC	

##################################
_pyGridderSmear.so:  GridderSmear.o 
	gcc $(CFLAGS) -fopenmp -pthread -shared GridderSmear.o -o _pyGridderSmear.so -lrt 

# ---- gcc C compile ------------------
GridderSmear.o:  GridderSmear.c GridderSmear.h
	gcc $(CFLAGS)  -lgomp -fopenmp -pthread -Ofast --fast-math  -c GridderSmear.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC -lrt
#	gcc $(CFLAGS)  -lgomp -fopenmp -pthread -O3 -msse -msse2 -msse3 -msse4.2 -mssse3 -march=corei7 -mtune=corei7 --fast-math  -c GridderSmear.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC -lrt
#	gcc $(CFLAGS)  -lgomp -fopenmp -pthread -Ofast --fast-math  -c GridderSmear.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC	


##################################
_pyGridderSmearPols.so:  GridderSmearPols.o 
	gcc $(CFLAGS) -fopenmp -pthread -shared GridderSmearPols.o -o _pyGridderSmearPols.so -lrt -L. -lTools

# ---- gcc C compile ------------------
GridderSmearPols.o:  GridderSmearPols.c GridderSmearPols.h Matrix.c JonesServer.c
	gcc $(CFLAGS)  -lgomp -fopenmp -pthread -Ofast --fast-math  -c GridderSmearPols.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC -lrt  -L. -lTools


##################################
_pyArrays.so:  Arrays.o 
	gcc $(CFLAGS) -fopenmp -pthread -shared Arrays.o -o _pyArrays.so -lrt 

Arrays.o:  Arrays.c Arrays.h
	gcc $(CFLAGS)  -lgomp -fopenmp -pthread  -O3 --fast-math  -c Arrays.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC -lrt


WeightingCore.c: WeightingCore.pyx
	cython $<
	
WeightingCore.so: WeightingCore.c
	gcc $(CFLAGS) -shared -pthread -fPIC -fwrapv -O2 -Wall -I/usr/include/python2.7 -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include -fno-strict-aliasing -o $@ $<





#gcc -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC

# # ---- Link --------------------------- 
# _pyGridder1D.so:  Gridder1D.o 
# 	gcc -shared Gridder1D.o -o _pyGridder1D.so

# # ---- gcc C compile ------------------
# Gridder1D.o:  Gridder1D.c Gridder1D.h
# 	gcc -c Gridder1D.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC

