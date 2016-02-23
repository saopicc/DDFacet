# DDFacet

## Dependencies

From an Ubuntu 14.04 base:

```
sudo pip install SharedArray
sudo pip install Polygon2
sudo pip install pyFFTW
sudo apt-get install python-casacore libfftw3-dev python-pyephem python-numexpr cython
```

Then need to clone or checkout the following three:

```
git clone git@github.com:cyriltasse/SkyModel.git
git clone git@github.com:cyriltasse/killMS2.git
git clone git@github.com:cyriltasse/DDFacet.git

```

## Build

Build a few libraries:

```
(cd DDFacet/Gridder ; make)
(cd ./killMS2/Predict ; make)
(cd ./killMS2/Predict ; make)
```

