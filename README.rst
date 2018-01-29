DDFacet
###################################
A facet-based radio imaging package
    .. image:: https://jenkins.meqtrees.net/job/DDFacet_master_cron/badge/icon
        :alt: Build status
        :target: https://jenkins.meqtrees.net/job/DDFacet_master_cron

    .. image:: https://img.shields.io/aur/license/yaourt.svg
        :alt: AUR

Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

(Users / Recommended - Docker based) Run via. Stimela >= 0.2.9 
==========================================================
We recommend running the imaging package through the Stimela framework <https://github.com/SpheMakh/Stimela>, built on a
widely supported containerization framework, called Docker. This package is on PiPY and and is purely python-based, requiring no dependencies other than Docker. It gives the user instantanious access to other commonly used packages such as Meqtrees, CASA, etc.


1. Install the latest docker from the Docker PPA: <https://docs.docker.com/engine/installation/linux/ubuntu/>. If you're not running Debian then select the suitable distribution. 

2. Ensure to add your user to the ``docker`` group. On Debian-based systems this can be done as follows::

        $ sudo usermod -aG docker $USER

3. Set up a virtual environment, activate it and upgrade pip, setuptools and wheel to the latest PyPI versions::

        $ virtualenv stimelavenv
        $ source stimelavenv/bin/activate
        (stimelavenv)$ pip install -U pip wheel setuptools
        (stimelavenv)$ pip install stimela

4. Run ``stimela pull`` and ``stimela build`` to pull all the latest astronomy software from DockerHub (this will take a while and is several GiB in size, so ensure you're on a fast link)::

        (stimelavenv)$ stimela pull
        (stimelavenv)$ stimela build

5. ``stimela cabs -i ddfacet`` lists all available options for the imager.

6. You can then add DDFacet as part of a larger reduction script, for example::

          1 import stimela
          2 
          3 INPUT="input"
          4 OUTPUT="output"
          5 MSDIR="msdir"
          6 
          7 recipe = stimela.Recipe("Test DDFacet imaging", ms_dir=MSDIR)
          8 # ...any other calibration steps here...
          9 recipe.add("cab/ddfacet", "ddfacet_test",
         10            {
         11                "Data-MS": ["3C147.MS/SUBMSS/D147-LO-NOIFS-NOPOL-4M5S.MS"],
         12                "Output-Name": "testimg",
         13                "Image-NPix": 2048,
         14                "Image-Cell": 2,
         15                "Cache-Reset": True,
         16                "Freq-NBand": 3,
         17                "Weight-ColName": "WEIGHT",
         18                "Beam-Model": "FITS",
         19                "Beam-FITSFile": "'beams/JVLA-L-centred_$(corr)_$(reim).fits'",
         20                "Data-ChunkHours": 0.5,
         21                "Data-Sort": True
         22            },
         23            input=INPUT, output=OUTPUT, shared_memory="14gb",
         24            label="test_image:: Make a test image using ddfacet")
         25 # ... any post imaging / additional calibration steps here ...
         26 recipe.run()

7. Run the script with::

        (stimelavenv)$ stimela run myscriptname.py

8. When you're done deactivate the virtual environment::

        (stimelavenv)$ deactivate

        
(Users / PyPI alternative) Virtual environment and pip:
==========================================================
We prefer that users use DDFacet though Docker. However, if this is not available (e.g. cluster
environments) we recommend you use a virtual environment. If you install it directly into your system packages you're
on your own -- be warned!!

1. You need to add in the KERN 3 ppa if you don't already have it::

        add-apt-repository -y -s ppa:kernsuite/kern-3

2. Install each of the debian dependencies. The latest full list of apt dependencies can be be found in the Dockerfile <https://github.com/saopicc/DDFacet/blob/master/Dockerfile>

3. Create a virtual environment somewhere on your system and activate::

        virtualenv --system-site-packages ddfacet
        source ddfacet/bin/activate
        
   Adding the `--system-site-packages` directive ensures that the virtualenv has access to system packages (such as meqtrees).
        
4. Then, install directly from the Python Package Index (PyPI) using pip - **ensure your venv is activated**::

        pip install -U pip setuptools
        pip install DDFacet --force-reinstall -U

5. When you're done with your imaging business::

        deactivate
        
(Users/Optional) Montblanc and pyMORESANE installation
==========================================================
Montblanc <https://github.com/ska-sa/montblanc> requires DDFacet to be installed in a virtual environment. **This section requires the DDFacet virtual environment to be activated and that you are in the DDFacet directory.**::
    
        (ddfvenv) $ pip install -r requirements.txt

(Users/Troubleshooting) Configure max shared memory
==========================================================
Running DDFacet on large images requires a lot of shared memory. Most systems limit the amount of shared memory to about 10%. To increase this limit add the following line to your ``/etc/default/tmpfs`` file::

        SHM_SIZE=100%

A restart will be required for this change to reflect. If you would prefer a once off solution execute the following line::

        sudo mount -o remount,size=100% /run/shm

It may also be necessary to run the following to remove the kernel security limit on mlock pinning. Without this things may
be slower than usual::

        echo "*        -   memlock     unlimited" > /etc/security/limits.conf

(Developers/Note): Architecture dependent binary
==========================================================
The default build system **DOES NOT** produce portable binaries at the cost of a slight improvement in runtime. You have to modify setup.cfg
and set the following lines in both [install] and [build] **before** compiling packages:

```
compopts=""
```

(Developers/Recommended): setting up your dev environment
==========================================================
**NOTE:Setup your virtual environment just as specified in the user section above. Ensure you activate!**

To setup your local development environment navigate clone DDFacet and run::

        (ddfvenv) $ git clone https://github.com/cyriltasse/DDFacet
        (ddfvenv) $ cd DDFacet
        (ddfvenv) $ git submodule update --init --recursive
        (ddfvenv) $ cd ..
        (ddfvenv) $ pip install -r DDFacet/requirements.txt
        (ddfvenv) $ pip install -e DDFacet/
        #To (re-)build the backend in your checked out folder:
        (ddfvenv) $ cd DDFacet
        (ddfvenv) $ python setup.py build

**IMPORTANT NOTE: You may need to remove the development version before running PIP when installing**

(Developers/Testing) Docker-based build
==========================================================
1. Simply pull the latest DDFacet and build the Docker image::

    git clone git@github.com:cyriltasse/DDFacet.git
    cd DDFacet
    docker build -t ddf .

2. You should now be able to run DDFacet in a container. Note that your parsets must have filenames relative to the mounted volume inside the container, for instance::

    docker run --shm-size 6g -v /scratch/TEST_DATA:/mnt ddf /mnt/test-master1.parset

**Important: if you ran ``git submodule update --init --recursive`` before you may need to remove the cached SkyModel before building the docker image with ``git rm --cached SkyModel``**

(Developers/Debugging) Build a few libraries (by hand with custom flags)
==========================================================
You can build against custom versions of libraries such is libPython and custom numpy versions.
To do this modify setup.cfg. Find and modify the following lines::

    compopts=-DENABLE_NATIVE_TUNING=ON -ENABLE_FAST_MATH=ON -DCMAKE_BUILD_TYPE=Release
    # or -DCMAKE_BUILD_TYPE=RelWithDebInfo for developers: this includes debugging symbols
    # or -DCMAKE_BUILD_TYPE=Debug to inspect the stacks using kdevelop or something similar

(Developers/Acceptance tests)
==========================================================
Paths
---------------------------------------------------------
Add this to your ``.bashrc``::

        export DDFACET_TEST_DATA_DIR=[folder where you keep the acceptance test data and images]
        export DDFACET_TEST_OUTPUT_DIR=[folder where you want the acceptance test output to be dumped]

To test your branch against the master branch using Jenkins
---------------------------------------------------------
Most of the core use cases will in the nearby future have reference images and an automated acceptance test.

Please **do not** commit against cyriltasse/master. The correct strategy is to branch/fork and do a pull request on Github
to merge changes into master. Once you opened a pull request add the following comment: "ok to test". This will let the Jenkins server know to start testing. You should see that the pull request and commit statusses shows "Pending". If the test succeeds you should see "All checks have passed" above the green merge button. Once the code is reviewed it will be merged into the master branch.

To run the tests on your local machine:
---------------------------------------------------------
You can run the automated tests by grabbing the latest set of measurements and reference images from the web and
extracting them to the directory you set up in your **DDFACET_TEST_DATA_DIR** environment variable. You can run
the automated tests by navigating to your DDFacet directory and running nosetests.

Each of the test cases is labeled by a class name and has reference images and a parset file with the same
name, ie. if the test case that has failed is called "TestWidefieldDirty" the reference images will be called the same. You should investigate the reason for any severe discrepancies between the output of the test case and the images produced by your changed codebase. See the docstring at the top of the class ClassCompareFITSImage for help and
filename conventions.

Acceptance test data can be found on the Jenkins server in the **/var/lib/jenkins/test-data** directory.

Adding more tests and creating new reference images.
---------------------------------------------------------
To resimulate images and add more tests:

In the Jenkins server data directory add a recipe to the makefile simulate and/or set up new reference images. This should only be done with the ``origin/master`` branch - not your branch or fork! Use the ddfacet-generate-refims task
to do this. You should manually verify that all the reference images are correct when you regenerate them. Each time you add a new option to DDFacet also add an option to the makefile in this directory. Once the option is set up in the makefile you can build the reference images on Jenkins.

Important directories on the CI server: 
---------------------------------------------------------
 - Reference data stored here: /var/lib/jenkins/test-data
 - /var/lib/jenkins/jobs/ddfacet-pr-build/workspace
 - /var/lib/jenkins/jobs/DDFacet_master_cron/workspace
 - /var/lib/jenkins/jobs/DDFacet_experimental/workspace


[tf_pip_install]: <https://www.tensorflow.org/get_started/os_setup#pip_installation>


