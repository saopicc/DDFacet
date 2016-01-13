all: _pyGridder.so _pyGridderSmear.so _pyGridderSmearPols.so _pyArrays.so

# ---- Link --------------------------- 
_pyGridder.so:  Gridder.o 
	gcc -fopenmp -pthread -shared -lrt Gridder.o -o _pyGridder.so

# ---- gcc C compile ------------------
Gridder.o:  Gridder.c Gridder.h
#	gcc -O3 -msse -msse2 -msse2avx -msse3 -msse4.2 -mssse3 -march=corei7 -mtune=corei7 -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC	

	gcc  -lgomp -fopenmp -pthread -O3 -msse -msse2 -msse3 -msse4.2 -mssse3 -march=corei7 -mtune=corei7 --fast-math -lrt  -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC
#	gcc  -lgomp -fopenmp -pthread -Ofast --fast-math  -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC	

##################################
_pyGridderSmear.so:  GridderSmear.o 
	gcc -fopenmp -pthread -shared GridderSmear.o -o _pyGridderSmear.so -lrt 

# ---- gcc C compile ------------------
GridderSmear.o:  GridderSmear.c GridderSmear.h
	gcc  -lgomp -fopenmp -pthread -Ofast --fast-math  -c GridderSmear.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC -lrt
#	gcc  -lgomp -fopenmp -pthread -O3 -msse -msse2 -msse3 -msse4.2 -mssse3 -march=corei7 -mtune=corei7 --fast-math  -c GridderSmear.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC -lrt
#	gcc  -lgomp -fopenmp -pthread -Ofast --fast-math  -c GridderSmear.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC	


##################################
_pyGridderSmearPols.so:  GridderSmearPols.o 
	gcc -fopenmp -pthread -shared GridderSmearPols.o -o _pyGridderSmearPols.so -lrt -L. -lTools

# ---- gcc C compile ------------------
GridderSmearPols.o:  GridderSmearPols.c GridderSmearPols.h Matrix.c JonesServer.c
	gcc  -lgomp -fopenmp -pthread -Ofast --fast-math  -c GridderSmearPols.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC -lrt  -L. -lTools


##################################
_pyArrays.so:  Arrays.o 
	gcc -fopenmp -pthread -shared Arrays.o -o _pyArrays.so -lrt 

Arrays.o:  Arrays.c Arrays.h
	gcc  -lgomp -fopenmp -pthread  -O3 --fast-math  -c Arrays.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC -lrt








#gcc -c Gridder.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC

# # ---- Link --------------------------- 
# _pyGridder1D.so:  Gridder1D.o 
# 	gcc -shared Gridder1D.o -o _pyGridder1D.so

# # ---- gcc C compile ------------------
# Gridder1D.o:  Gridder1D.c Gridder1D.h
# 	gcc -c Gridder1D.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC

