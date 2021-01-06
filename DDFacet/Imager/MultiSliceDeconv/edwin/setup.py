# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name='edwin',
    version='0.2.0',
    author='François Orieux',
    author_email='orieux@lss.supelec.fr',
    packages=['edwin'],
    scripts=[],
    url='http://bitbucket.org/forieux/edwin/',
    license='LICENSE.txt',
    description='Module related to Bayesian approach to inverse problems',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy >= 1.4",
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
    ],
)
