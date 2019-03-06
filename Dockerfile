#From Ubuntu 18.04
FROM kernsuite/base:5
MAINTAINER Ben Hugo "bhugo@ska.ac.za"

#Package dependencies
COPY apt.sources.list /etc/apt/sources.list

#Setup environment
ENV DDFACET_TEST_DATA_DIR /test_data
ENV DDFACET_TEST_OUTPUT_DIR /test_output
ENV DEBIAN_FRONTEND noninteractive

# Support large mlocks
RUN echo "*        -   memlock     unlimited" > /etc/security/limits.conf
ENV DEB_SETUP_DEPENDENCIES \
    dpkg-dev \
    g++ \
    gcc \
    libc-dev \
    cmake \
    gfortran \
    git \
    wget \
    subversion

ENV DEB_DEPENCENDIES \
    python-pip \
    libfftw3-dev \
    casacore-data \
    casacore-dev \
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
    meqtrees* \
    python-owlcat \
    owlcat \
    tigger-lsm \
    # LOFAR Beam and including makems needed for ref image generation
    lofarbeam-dev \
    makems \
    # Reference image generation dependencies
    make

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y -s ppa:kernsuite/kern-5
RUN apt-add-repository -y multiverse
RUN apt-get update
RUN apt-get install -y $DEB_SETUP_DEPENDENCIES
RUN apt-get install -y $DEB_DEPENCENDIES

# Copy DDFacet and SkyModel into the image
ADD DDFacet /src/DDFacet/DDFacet
ADD SkyModel /src/DDFacet/SkyModel
ADD MANIFEST.in /src/DDFacet/MANIFEST.in
ADD requirements.txt /src/DDFacet/requirements.txt
ADD setup.py /src/DDFacet/setup.py
ADD setup.cfg /src/DDFacet/setup.cfg
ADD README.rst /src/DDFacet/README.rst
ADD .git /src/DDFacet/.git
ADD .gitignore /src/DDFacet/.gitignore
ADD .gitmodules /src/DDFacet/.gitmodules

# Start install
RUN pip install -U pip setuptools wheel
RUN pip install -U "astropy == 2.0.11"
RUN pip install -U "numpy <= 1.16.1"
# Install Montblanc and all other optional dependencies
RUN pip install -r /src/DDFacet/requirements.txt
RUN cd /src/DDFacet/ && git submodule update --init --recursive && cd /
# Finally install DDFacet
RUN rm -rf /src/DDFacet/DDFacet/cbuild
RUN pip install -I -U /src/DDFacet/
# Nuke the unused & cached binaries needed for compilation, etc.
RUN rm -r /src/DDFacet
RUN apt-get remove -y $DEB_SETUP_DEPENDENCIES
RUN apt-get autoclean -y
RUN apt-get clean -y
RUN apt-get autoremove -y
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache/
RUN rm -rf /var/cache/
RUN rm -rf LOFAR-Release-2_21_9

# Set MeqTrees Cattery path to virtualenv installation directory
ENV MEQTREES_CATTERY_PATH /usr/lib/python2.7/dist-packages/Cattery/
# basic test of installation
RUN DDF.py --help
RUN fitstool.py --help
ENTRYPOINT ["DDF.py"]
CMD ["--help"]
