#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This file contains the python (distutils/setuptools) instructions so your
# package can be installed on **any** host system. It defines some basic
# information like the package name for instance, or its homepage.
#
# It also defines which other packages this python package depends on and that
# are required for this package's operation. The python subsystem will make
# sure all dependent packages are installed or will install them for you upon
# the installation of this package.
#
# The 'buildout' system we use here will go further and wrap this package in
# such a way to create an isolated python working environment. Buildout will
# make sure that dependencies which are not yet installed do get installed, but
# **without** requiring adminstrative privileges on the host system. This
# allows you to test your package with new python dependencies w/o requiring
# administrative interventions.

bob_packages = ['bob.core', 'bob.io.base', 'bob.sp', 'bob.ip.base']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension', 'bob.blitz'] + bob_packages))
from bob.extension.utils import egrep, find_header, find_library
from bob.blitz.extension import Extension, build_ext

from bob.extension.utils import load_requirements
build_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name='bob.ip.facedetect',
    version=version,
    description='Face detection using boosted LBP features',

    url='http://gitlab.idiap.ch/bob/bob.ip.facedetect',
    license='GPLv3',
    author='Manuel Guenther',
    author_email='manuel.guenther@idiap.ch',
    keywords='bob, facereclib, face detection',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need adminstrative
    # privileges when using buildout.
    setup_requires = build_requires,
    install_requires = build_requires,

    cmdclass={
      'build_ext': build_ext,
    },

    ext_modules = [
      Extension("bob.ip.facedetect.version",
        [
          "bob/ip/facedetect/version.cpp",
        ],
        version = version,
        bob_packages = bob_packages,
      ),

      Extension(
        'bob.ip.facedetect._library',
        [
          "bob/ip/facedetect/cpp/features.cpp",
          "bob/ip/facedetect/cpp/boundingbox.cpp",

          "bob/ip/facedetect/bounding_box.cpp",
          "bob/ip/facedetect/feature_extractor.cpp",
          "bob/ip/facedetect/main.cpp",
        ],
        version = version,
        bob_packages = bob_packages,
      )
    ],

    # Your project should be called something like 'xbob.<foo>' or
    # 'xbob.<foo>.<bar>'. To implement this correctly and still get all your
    # packages to be imported w/o problems, you need to implement namespaces
    # on the various levels of the package and declare them here. See more
    # about this here:
    # http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
    #
    # Our database packages are good examples of namespace implementations
    # using several layers. You can check them out here:
    # https://gitlab.idiap.ch/bob/bob/wikis/Packages


    # This entry defines which scripts you will have inside the 'bin' directory
    # once you install the package (or run 'bin/buildout'). The order of each
    # entry under 'console_scripts' is like this:
    #   script-name-at-bin-directory = module.at.your.library:function
    #
    # The module.at.your.library is the python file within your library, using
    # the python syntax for directories (i.e., a '.' instead of '/' or '\').
    # This syntax also omits the '.py' extension of the filename. So, a file
    # installed under 'example/foo.py' that contains a function which
    # implements the 'main()' function of particular script you want to have
    # should be referred as 'example.foo:main'.
    #
    # In this simple example we will create a single program that will print
    # the version of bob.
    entry_points={
      # scripts should be declared using this entry:
      'console_scripts': [
        'collect_training_data.py = bob.ip.facedetect.script.collect_training_data:main',
        'extract_training_features.py = bob.ip.facedetect.script.extract_training_features:main',
        'train_detector.py = bob.ip.facedetect.script.train_detector:main',
        'validate_detector.py = bob.ip.facedetect.script.validate_detector:main',
        'detect_faces.py = bob.ip.facedetect.script.detect_faces:main',
        'evaluate_detections.py = bob.ip.facedetect.script.evaluate:main',
        'plot_froc.py = bob.ip.facedetect.script.plot_froc:main'
      ],
    },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      'Environment :: Plugins',
    ],
)
