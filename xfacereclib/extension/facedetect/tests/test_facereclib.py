#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import unittest
import os
import numpy
import facereclib
from nose.plugins.skip import SkipTest

import pkg_resources

import bob.io.base
import bob.io.base.test_utils

regenerate_refs = False

class DetectionTest (unittest.TestCase):

  def execute(self, preprocessor, data, annotations, reference):
    # execute the preprocessor
    preprocessed = preprocessor(data, annotations)
    reference_file = pkg_resources.resource_filename('xfacereclib.extension.facedetect', os.path.join('tests', reference))
    if regenerate_refs:
      bob.io.base.save(preprocessed, reference_file)

    self.assertTrue((numpy.abs(bob.io.base.load(reference_file) - preprocessed) < 1e-5).all())

  def test01_face_crop(self):
    # read input
    test_image = bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'facereclib', 'tests'))
    preprocessor = facereclib.utils.tests.load_resource('face-detect', 'preprocessor')
    # execute preprocessor
    self.execute(preprocessor, test_image, None, 'detected.hdf5')

  def test02_tan_triggs(self):
    # read input
    test_image = bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'facereclib', 'tests'))
    preprocessor = facereclib.utils.tests.load_resource('face-detect+tan-triggs', 'preprocessor')
    # execute preprocessor
    self.execute(preprocessor, test_image, None, 'detected+tt.hdf5')

  def test03_inorm_lbp(self):
    # read input
    test_image = bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'facereclib', 'tests'))
    preprocessor = facereclib.utils.tests.load_resource('face-detect+inorm-lbp', 'preprocessor')
    # execute preprocessor
    self.execute(preprocessor, test_image, None, 'detected+lbp.hdf5')

  def test04_flandmark(self):
    # read input
    test_image = bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'facereclib', 'tests'))
    preprocessor = facereclib.utils.tests.load_resource('face-detect-flandmark', 'preprocessor')
    # execute preprocessor
    self.execute(preprocessor, test_image, None, 'detected+flandmark.hdf5')




