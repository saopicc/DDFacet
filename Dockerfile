#From Ubuntu 18.04
FROM ubuntu:18.04
MAINTAINER Ben Hugo "bhugo@ska.ac.za"

#Package dependencies
COPY apt.sources.list /etc/apt/sources.list

#Setup environment
ENV DDFACET_TEST_DATA_DIR /test_data
ENV DDFACET_TEST_OUTPUT_DIR /test_output

# Support large mlocks
RUN echo "*        -   memlock     unlimited" > /etc/security/limits.conf
ENV DEBIAN_FRONTEND noninteractive
ENV DEBIAN_PRIORITY critical
ENV GNUCOMPILER 7
ENV DEB_SETUP_DEPENDENCIES \
    dpkg-dev \
    g++-$GNUCOMPILER \
    gcc-$GNUCOMPILER \
    libc-dev \
    cmake \
    gfortran-$GNUCOMPILER \
    git \
    wget \
    subversion

ENV DEB_DEPENCENDIES \
    python-pip \
    libfftw3-dev \
    python-numpy \
    libfreetype6 \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    python2.7-dev \
    libboost-all-dev \
    libcfitsio-dev \
    libhdf5-dev \
    wcslib-dev \
    libatlas-base-dev \
    liblapack-dev \
    python-tk \
    libreadline6-dev \
    subversion \
    liblog4cplus-dev \
    libhdf5-dev \
    libncurses5-dev \
    libsofa1-dev \
    flex \
    bison \
    libbison-dev \
    # Reference image generation dependencies
    make

RUN apt-get update
RUN apt-get install -y $DEB_SETUP_DEPENDENCIES
RUN apt-get install -y $DEB_DEPENCENDIES

ENV PATH /usr/local/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH /usr/local/lib/python2.7/site-packages:$PYTHONPATH

# Latest GCC segfaults when compiling casacore
RUN rm /usr/bin/gcc /usr/bin/g++ /usr/bin/cpp /usr/bin/cc
RUN ln -s /usr/bin/gcc-$GNUCOMPILER /usr/bin/gcc
RUN ln -s /usr/bin/g++-$GNUCOMPILER /usr/bin/g++
RUN ln -s /usr/bin/gcc-$GNUCOMPILER /usr/bin/cc
RUN ln -s /usr/bin/g++-$GNUCOMPILER /usr/bin/cpp
RUN ln -s /usr/bin/gfortran-$GNUCOMPILER /usr/bin/gfortran

#####################################################################
## BUILD CASACORE FROM SOURCE
#####################################################################
RUN mkdir /src
WORKDIR /src
RUN wget https://github.com/casacore/casacore/archive/v3.1.1.tar.gz
RUN tar xvf v3.1.1.tar.gz
RUN mkdir casacore-3.1.1/build
WORKDIR /src/casacore-3.1.1/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release -DBUILD_DEPRECATED=ON -DBUILD_PYTHON=ON -DBUILD_PYTHON3=ON ../
RUN make -j 4
RUN make install
RUN ldconfig
#RUN pip install -U --user --force-reinstall --install-option="--prefix=/usr"  pip setuptools wheel
WORKDIR /src
RUN rm v3.1.1.tar.gz
RUN wget https://github.com/casacore/python-casacore/archive/v3.1.1.tar.gz
RUN tar xvf v3.1.1.tar.gz
WORKDIR /src/python-casacore-3.1.1
RUN pip install .
WORKDIR /
RUN python -c "from pyrap.tables import table as tbl"

#####################################################################
## Get CASACORE ephem data
#####################################################################
RUN mkdir -p /usr/share/casacore/data/
WORKDIR /usr/share/casacore/data/
RUN apt-get install -y rsync
RUN rsync -avz rsync://casa-rsync.nrao.edu/casa-data .

#####################################################################
## BUILD MAKEMS FROM SOURCE AND TEST
#####################################################################
RUN mkdir -p /src/
WORKDIR /src
ENV BUILD /src
RUN wget https://github.com/ska-sa/makems/archive/1.5.3.tar.gz
RUN tar xvf 1.5.3.tar.gz
RUN mkdir -p $BUILD/makems-1.5.3/LOFAR/build/gnu_opt
WORKDIR $BUILD/makems-1.5.3/LOFAR/build/gnu_opt
RUN cmake -DCMAKE_MODULE_PATH:PATH=$BUILD/makems-1.5.3/LOFAR/CMake \
-DUSE_LOG4CPLUS=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release ../..
RUN make -j 16
RUN make install

ENV PATH=/src/makems-1.5.3/LOFAR/build/gnu_opt/CEP/MS/src:${PATH}
WORKDIR $BUILD/makems-1.5.3/test
RUN makems WSRT_makems.cfg

#####################################################################
## BUILD CASArest from source
#####################################################################
WORKDIR /src
RUN git clone https://github.com/casacore/casarest
WORKDIR /src/casarest
RUN mkdir -p build
WORKDIR /src/casarest/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release ../
RUN make -j 32
RUN make install
RUN ldconfig

#####################################################################
## BUILD MEQTREES FROM SOURCE AND TEST
#####################################################################
# Owlcat
WORKDIR /src
RUN wget https://github.com/ska-sa/owlcat/archive/v1.6.0.tar.gz
RUN tar -xvf v1.6.0.tar.gz
WORKDIR /src/owlcat-1.6.0
RUN pip install .

# kittens
WORKDIR /src
RUN wget https://github.com/ska-sa/kittens/archive/1.4.3.tar.gz
RUN tar -xvf 1.4.3.tar.gz
WORKDIR /src/kittens-1.4.3
RUN pip install .

# purr
WORKDIR /src
RUN wget https://github.com/ska-sa/purr/archive/v1.5.0.tar.gz
RUN tar -xvf v1.5.0.tar.gz
WORKDIR /src/purr-1.5.0
RUN pip install .

# tigger-lsm
WORKDIR /src
RUN rm v1.6.0.tar.gz
RUN wget https://github.com/ska-sa/tigger-lsm/archive/v1.6.0.tar.gz
RUN tar -xvf v1.6.0.tar.gz
WORKDIR /src/tigger-lsm-1.6.0
RUN pip install .

# Cattery
WORKDIR /src
RUN wget https://github.com/ska-sa/meqtrees-cattery/archive/v1.7.0.tar.gz
RUN tar -xvf v1.7.0.tar.gz
WORKDIR /src/meqtrees-cattery-1.7.0
RUN pip install .

# blitz
WORKDIR /src
RUN wget https://github.com/blitzpp/blitz/archive/1.0.1.tar.gz
RUN tar -xvf 1.0.1.tar.gz
WORKDIR /src/blitz-1.0.1
RUN ./configure --enable-shared
RUN make -j16
RUN make install
RUN ldconfig

# timba
RUN apt-get install -y libqdbm-dev
WORKDIR /src
RUN wget https://github.com/ska-sa/meqtrees-timba/archive/v1.7.0.tar.gz
RUN tar -xvf v1.7.0.tar.gz.1
RUN mkdir /src/meqtrees-timba-1.7.0/build
WORKDIR /src/meqtrees-timba-1.7.0/build
RUN cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_PYTHON_3=OFF ..
RUN make -j4
RUN make install
RUN ldconfig

# get the test from pyxis
WORKDIR /src
RUN rm v1.7.0.tar.gz
RUN wget https://github.com/ska-sa/pyxis/archive/v1.7.0.tar.gz
RUN tar -xvf v1.7.0.tar.gz
WORKDIR /src/pyxis-1.7.0
RUN pip install .
RUN cp -r /src/pyxis-1.7.0/Pyxis/recipies /usr/local/lib/python2.7/dist-packages/Pyxis/recipies

# run test when built
RUN apt-get install -y python-qt4
RUN pip install nose
WORKDIR /usr/local/lib/python2.7/dist-packages/Pyxis/recipies/meqtrees-batch-test
RUN python2.7 -m "nose"

#####################################################################
## BUILD LOFAR FROM SOURCE
#####################################################################
WORKDIR /src
RUN git clone https://github.com/lofar-astron/LOFARBeam.git
WORKDIR LOFARBeam
RUN git checkout db8d082b34c203417b01d20b6f3bf0df17728ee8
#RUN svn co --non-interactive --no-auth-cache --username lofar-guest --password lofar-guest https://svn.astron.nl/LOFAR/tags/LOFAR-Release-4_0_9
#WORKDIR LOFAR-Release-4_0_9
#RUN wget https://codeload.github.com/lofar-astron/LOFARBeam/tar.gz/v4.0
#RUN tar xvf v4.0
RUN mkdir -p build/gnucxx11_opt
WORKDIR build/gnucxx11_opt
RUN cmake -DBUILD_PACKAGES="pystationresponse" -DCMAKE_INSTALL_PREFIX=/usr ../../
RUN make -j16
RUN make install
WORKDIR /
RUN touch /usr/lib/python2.7/site-packages/lofar/__init__.py
ENV PYTHONPATH "/usr/lib/python2.7/site-packages:$PYTHONPATH"
RUN python -c "import lofar.stationresponse as lsr"

RUN apt-get update
RUN apt-get install -y gfortran
RUN pip install "bdsf<=1.8.15"
RUN python -c "import bdsf"

#####################################################################
## BUILD DDF FROM SOURCE
#####################################################################
#Copy DDFacet and SkyModel into the image
ADD DDFacet /src/DDFacet/DDFacet
ADD SkyModel /src/DDFacet/SkyModel
ADD MANIFEST.in /src/DDFacet/MANIFEST.in
ADD setup.py /src/DDFacet/setup.py
ADD setup.cfg /src/DDFacet/setup.cfg
ADD README.rst /src/DDFacet/README.rst
ADD pyproject.toml /src/DDFacet/pyproject.toml
ADD .git /src/DDFacet/.git
ADD .gitignore /src/DDFacet/.gitignore
ADD .gitmodules /src/DDFacet/.gitmodules

RUN cd /src/DDFacet/ && git submodule update --init --recursive && cd /
# Finally install DDFacet
RUN rm -rf /src/DDFacet/DDFacet/cbuild
RUN pip install -U pip setuptools wheel
RUN python2.7 -m pip install pybind11
RUN python2.7 -m pip install -U "/src/DDFacet/[dft-support,moresane-support,testing-requirements,fits-beam-support]"
RUN cd /src/DDFacet/ && python2.7 setup.py build && cd /
# Set MeqTrees Cattery path to installation directory
ENV MEQTREES_CATTERY_PATH /usr/local/lib/python2.7/dist-packages/Cattery/
ENV PYTHONPATH $MEQTREES_CATTERY_PATH:$PYTHONPATH

RUN pip install -U astropy==2.0.10 # temporary py2 kludge to work around a WCS segfault

RUN python2.7 -c "import Siamese"

# perform some basic tests
RUN DDF.py --help
RUN MakeMask.py --help
RUN MakeCatalog.py --help
RUN MakeModel.py --help
RUN MaskDicoModel.py --help
RUN ClusterCat.py --help

# set as entrypoint - user should be able to run docker run ddftag and get help printed
ENTRYPOINT ["DDF.py"]
CMD ["--help"]
