FROM radioastro/base
MAINTAINER Ben Hugo "bhugo@ska.ac.za"
#Package dependencies\n\
COPY apt.sources.list /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y -s ppa:radio-astro/main
RUN apt-get update
RUN apt-get install -y git build-essential
RUN apt-get install -y python-pip
RUN apt-get install -y libfftw3-dev
RUN apt-get install -y cmake
RUN apt-get install -y casacore-data
RUN apt-get install -y libcasacore2-dev
RUN apt-get install -y libcasacore2
RUN apt-get install -y python-numpy
RUN apt-get install -y libfreetype6
RUN apt-get install -y libfreetype6-dev
RUN apt-get install -y libpng12.0
RUN apt-get install -y libpng12-dev
RUN apt-get install -y pkg-config
RUN apt-get install -y python2.7-dev
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libcfitsio3-dev
RUN apt-get install -y libatlas-dev
RUN apt-get install -y gfortran
RUN apt-get install -y libatlas-dev
RUN apt-get install -y liblapack-dev
RUN apt-get install -y python-tk
RUN apt-get install -y meqtrees-timba
#Reference image generation required packages
RUN apt-get install -y meqtrees
RUN apt-get install -y makems
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
# Upgrade pip
RUN pip install -U pip
RUN pip install -U virtualenv
RUN pip install -U setuptools
# Setup DDFacet Virtual Environment
RUN virtualenv --system-site-packages /ddfvenv
#Install DDFacet
RUN cd /src/DDFacet/ ; git submodule update --init --recursive
RUN . /ddfvenv/bin/activate && pip install -I --force-reinstall /src/DDFacet/
# Install tensorflow CPU nightly
RUN . /ddfvenv/bin/activate && pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
# Clone montblanc and checkout the tensorflow implementation
RUN git clone https://github.com/ska-sa/montblanc.git /montblanc/; cd /montblanc/; git checkout 339eb8f8a0f4a44243f340b7f33882fd9656858b
# Make the tensorflow ops
RUN cd /montblanc/montblanc/impl/rime/tensorflow/rime_ops; . /ddfvenv/bin/activate && make -j 8
# Install montblanc in development mode
RUN cd /montblanc; . /ddfvenv/bin/activate && python setup.py develop
# Set MeqTrees Cattery path to virtualenv installation directory
ENV MEQTREES_CATTERY_PATH /ddfvenv/lib/python2.7/site-packages/Cattery/
# Execute virtual environment version of DDFacet
ENTRYPOINT ["/ddfvenv/bin/DDF.py"]
CMD ["--help"]
