FROM radioastro/base
MAINTAINER Ben Hugo "bhugo@ska.ac.za"

#Package dependencies
COPY apt.sources.list /etc/apt/sources.list

#Setup environment
ENV DDFACET_TEST_DATA_DIR /test_data
ENV DDFACET_TEST_OUTPUT_DIR /test_output

#Copy DDFacet and SkyModel into the image
ADD DDFacet /src/DDFacet/DDFacet
ADD SkyModel /src/DDFacet/SkyModel
ADD MANIFEST.in /src/DDFacet/MANIFEST.in
ADD requirements.txt /src/DDFacet/requirements.txt
ADD setup.py /src/DDFacet/setup.py
ADD README.md /src/DDFacet/README.md
ADD .git /src/DDFacet/.git
ADD .gitignore /src/DDFacet/.gitignore
ADD .gitmodules /src/DDFacet/.gitmodules

# Support large mlocks
RUN echo "*        -   memlock     unlimited" > /etc/security/limits.conf
ENV DEB_SETUP_DEPENDENCIES \
    dpkg-dev \
    g++ \
    gcc \
    libc-dev \
    cmake \
    gfortran \
    git

ENV DEB_DEPENCENDIES \
    python-pip \
    libfftw3-dev \
    casacore-data \
    libcasacore2-dev \
    libcasacore2-dev \
    libcasacore2 \
    python-numpy \
    libfreetype6 \
    libfreetype6-dev \
    libpng12.0 \
    libpng12-dev \
    pkg-config \
    python2.7-dev \
    libboost-all-dev \
    libcfitsio3-dev \
    libatlas-dev \
    libatlas-dev \
    liblapack-dev \
    python-tk \
    meqtrees-timba \
    # Reference image generation dependencies
    meqtrees \
    makems \
    make

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y -s ppa:radio-astro/main && \
    apt-get update && \
    apt-get install -y $DEB_SETUP_DEPENDENCIES && \
    apt-get install -y $DEB_DEPENCENDIES && \
    apt-get install -y git && \
    pip install -U pip virtualenv setuptools wheel && \
    virtualenv --system-site-packages /ddfvenv && \
    # Install DDFacet
    cd /src/DDFacet/ && git submodule update --init --recursive && cd / && \
    . /ddfvenv/bin/activate ; pip install -I --force-reinstall --no-binary :all: /src/DDFacet/ && \
    # Install montblanc
    . /ddfvenv/bin/activate ; pip install git+git://github.com/ska-sa/montblanc.git@65ffb611f5376380cbed0e76624b25581e9f4e4d && \
    # Nuke the unused & cached binaries needed for compilation, etc.
    apt-get remove -y $DEB_SETUP_DEPENDENCIES && \
    apt-get autoclean -y && \
    apt-get clean -y && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/ && \
    rm -rf /var/cache/

# Set MeqTrees Cattery path to virtualenv installation directory
ENV MEQTREES_CATTERY_PATH /ddfvenv/lib/python2.7/site-packages/Cattery/

# Execute virtual environment version of DDFacet
ENTRYPOINT ["/ddfvenv/bin/DDF.py"]
CMD ["--help"]