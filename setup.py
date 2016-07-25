
#! /usr/bin/env python
#
# Copyright (C) 2015-2016 Brahim Belaoucha <Brahim.Belaoucha@inria.fr>



descr = """dMRIParcellation: a package to parcellate the cortical surface using dMRI tractograms"""

import os
from setuptools import setup
from os import path as op 
from numpy.distutils.core import setup
version='0.0.0.dev'
DISTNAME="dMRIParcellation"
DESCRIPTION=descr
MAINTAINER='Brahim Belaoucha'
MAINTAINER_EMAIL='Brahim.Belaoucha@gmail.com'
LICENSE=''
URL='https://github.com/BBELAOUCHA/dMRIParcellation'
DOWNLOAD_URL='https://github.com/BBELAOUCHA/dMRIParcellation'
VERSION=version
import sys


if __name__ == "__main__":

    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        url=URL,
        download_url=DOWNLOAD_URL,
        packages=['inc','test'],
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'Operating System :: Fedora']
    )
