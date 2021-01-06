=========================
Edwin, bayesian inversion
=========================

The Bayesian inversion (``edwin``) package provides algorithm
developed during my scientific research work in numerical computation
for inverse problems (signal, image processing). Feel free to use them
as you want. Any comments and contributions are welcome.

The name ``edwin`` is in reference to Edwin T. Jaynes, a great
Bayesian Analysis scientific.

note : this README is relatively out of date.

Acknowledgements
================

The use of ``edwin`` software package should be explicitly
acknowledged in publications in the following form:

1. an acknowledgment statement: "Some of the results in this paper
   have been derived using some of the ``edwin`` package algorithms
   From F. Orieux et al. published in *citations*.

2. at the first reference, a footnote placed in the main body of the
   paper referring to the ``edwin`` web site, currently
   http://bitbucket.org/forieux/edwin

The citations are mentioned in documentation, *References* section of
this file and are available in bibtex file.

Info
====

* Author: François Orieux
* Contact: orieux at iap dot fr
* Project homepage: http://bitbucket.org/forieux/edwin
* Downloads page: https://bitbucket.org/forieux/edwin/downloads

Contents
========

improcessing.py
    A module that implement the algorithm described in [2] for
    unsupervised myopic image deconvolution. However the myopic part
    is not actually available.

inversion.py
    A module that implement the algorithm described in [1] and use in
    [3-4] and other papers. It's implement an unsupervised general
    inverse problem algorithm estimation, based on MCMC algorithm.

sampling.py
    Implementation of stochastic sampling algorithm, specially [1].

optim.py
    A module that implement classical optimisation algorithm for use
    of other module. They are design for very large system resolution
    (dim > 1e6).


Requirements
============

This package depends on my free otb package (utility functions).

* Numpy version >= 1.4.1

Installation
============

The ``pip`` version::

    pip install edwin

If you have not ``pip``, download the archive, decompress it and to
install in your user path, run in a command line::

    python setup.py install --user

or for the system path, run as root::

    python setup.py install

Development
===========

This package follow the Semantic Versionning convention
http://semver.org/. To get the development version you can clone the
mercurial repository available here
http://bitbucket.org/forieux/edwin

The ongoing development depends on my research activity but is open. I
try to fix bugs.

License
=======

``edwin`` is free software distributed under the MIT license, see
LICENSE.txt

References
==========

A bibtex file is provided in the archive.

.. [1] F. Orieux, O. Féron and J.-F. Giovannelli, "Sampling
   high-dimensional Gaussian distributions for general linear inverse
   problems", IEEE Signal Processing Letters, 2012

.. [2] François Orieux, Jean-François Giovannelli, and Thomas
   Rodet, "Bayesian estimation of regularization and point spread
   function parameters for Wiener-Hunt deconvolution",
   J. Opt. Soc. Am. A 27, 1593-1607 (2010)

.. [3] F. Orieux, E. Sepulveda, V. Loriette, B. Dubertret and
   J.-C. Olivo-Marin, "Bayesian Estimation for Optimized Structured
   Illumination Microscopy", IEEE trans. on Image Processing. 2012

.. [4] F. Orieux, J.-F. Giovannelli, T. Rodet, and A. Abergel,
   "Estimating hyperparameters and instrument parameters in
   regularized inversion Illustration for Herschel/SPIRE map
   making", Astronomy & Astrohpysics, 2013
