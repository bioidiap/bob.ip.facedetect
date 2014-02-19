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

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires='xbob.extension'))
from xbob.extension import Extension, build_ext

ext_debug = False

if ext_debug:
  ext_arguments = {
    'extra_compile_args' : [
      '-ggdb',
       '-std=c++11',
    ],
    'define_macros' : [
      ('BZ_DEBUG', 1)
    ],
   'undef_macros' : [
     'NDEBUG'
    ]
  }
else:
  ext_arguments = {
    'extra_compile_args' : [
       '-std=c++11',
    ],
  }

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name='xfacereclib.extension.facedetect',
    version='0.1',
    description='Example for using Bob inside a buildout project',

    url='http://github.com/idiap/bob.project.example',
    license='GPLv3',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',
    keywords='bob, xbob',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,

    setup_requires=[
      'xbob.extension',
    ],

    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need adminstrative
    # privileges when using buildout.
    install_requires=[
      'setuptools',
      'bob', # base signal proc./machine learning library
      'facereclib',
      'xbob.boosting',
      'xbob.flandmark'
    ],

    cmdclass={
      'build_ext': build_ext,
    },

    ext_modules = [
      Extension(
        'xfacereclib.extension.facedetect._features',
        [
          "xfacereclib/extension/facedetect/cpp/features.cpp",
          "xfacereclib/extension/facedetect/cpp/boundingbox.cpp",
          "xfacereclib/extension/facedetect/cpp/bindings.cpp",
        ],
        pkgconfig = [
          'bob-ip',
        ],
        include_dirs = [
          "/idiap/user/mguenther/Bob/release/include",
#          "xfacereclib/extension/facedetect/cpp"
        ],
# STUFF for DEBUGGING goes here (requires DEBUG bob version...):
        **ext_arguments
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
    # https://github.com/idiap/bob/wiki/Satellite-Packages
    namespace_packages = [
      'xfacereclib',
      'xfacereclib.extension',
    ],

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
        'train_detector.py = xfacereclib.extension.facedetect.script.train:main',
        'train_localizer.py = xfacereclib.extension.facedetect.script.train_localizer:main',
        'validate.py = xfacereclib.extension.facedetect.script.validate:main',
        'display.py = xfacereclib.extension.facedetect.script.display:main',
        'detect.py = xfacereclib.extension.facedetect.script.detect:main',
        'localize.py = xfacereclib.extension.facedetect.script.localize:main',
        'plots.py = xfacereclib.extension.facedetect.script.evaluate:main',
        'error.py = xfacereclib.extension.facedetect.script.errors:main',
      ],

      # registered database short cuts
      'facereclib.database': [
        'banca-french      = xfacereclib.extension.facedetect.databases.banca_french:database',
        'banca-spanish     = xfacereclib.extension.facedetect.databases.banca_spanish:database',
        'bioid             = xfacereclib.extension.facedetect.databases.bioid:database',
        'caltech           = xfacereclib.extension.facedetect.databases.caltech:database',
        'cinema            = xfacereclib.extension.facedetect.databases.cinema:database',
        'mit-cmu           = xfacereclib.extension.facedetect.databases.cmu:database',
        'fddb              = xfacereclib.extension.facedetect.databases.fddb:database',
        'fdhd              = xfacereclib.extension.facedetect.databases.fdhd:database',
        'mash              = xfacereclib.extension.facedetect.databases.mash:database',
        'ofd               = xfacereclib.extension.facedetect.databases.ofd:database',
        'cmu-pie           = xfacereclib.extension.facedetect.databases.pie:database',
        'yale-b            = xfacereclib.extension.facedetect.databases.yale_b:database',
        'web               = xfacereclib.extension.facedetect.databases.web:database',
        'multipie          = xfacereclib.extension.facedetect.databases.multipie:database',
      ],

      # registered preprocessors
      'facereclib.preprocessor': [
        'face-detect             = xfacereclib.extension.facedetect.configurations.face_crop:preprocessor',
        'face-detect+tan-triggs  = xfacereclib.extension.facedetect.configurations.tan_triggs:preprocessor',
        'face-detect+inorm-lbp   = xfacereclib.extension.facedetect.configurations.inorm_lbp:preprocessor',
        'face-detect-flandmark   = xfacereclib.extension.facedetect.configurations.flandmark:preprocessor',
      ],

      # tests that are _exported_ (that can be executed by other packages) can
      # be signalized like this:
      'bob.test': [
         'facedetect = xfacereclib.extension.facedetect.tests.test_facereclib:DetectionTest'
      ],

      # finally, if you are writing a database package, you must declare the
      # relevant entries like this:
      #'bob.db': [
      #  'example = xbob.example.driver:Interface',
      #  ]
      # Note: this is just an example, this package does not declare databases
      },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers = [
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
)
