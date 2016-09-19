FROM radioastro/base
MAINTAINER Ben Hugo "bhugo@ska.ac.za"
#Package dependencies\n\
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y -s ppa:radio-astro/main
RUN apt-get update
RUN apt-get install -y git build-essential python-pip libfftw3-dev cython cmake casacore-data libcasacore2-dev libcasacore2
#Setup environment
ENV DDFACET_TEST_DATA_DIR /WORKSPACE
ENV DDFACET_TEST_OUTPUT_DIR /WORKSPACE
#Copy DDFacet and SkyModel into the image
ADD DDFacet /src/DDFacet/DDFacet
ADD MANIFEST.in /src/DDFacet/MANIFEST.in
ADD requirements.txt /src/DDFacet/requirements.txt
ADD setup.py /src/DDFacet/setup.py
ADD README.md /src/DDFacet/README.md
ADD .git /src/DDFacet/.git
#Install DDFacet
RUN cd /src/DDFacet/ ; git rm --cached SkyModel ; git submodule update --init --recursive
RUN pip install /src/DDFacet/
#Pass any environment variables down to DDFacet
ENTRYPOINT "DDF.py"
