#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <manuel.guenther@idiap.ch>
# Tue Jul 2 14:52:49 CEST 2013
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

from __future__ import print_function

"""This script evaluates the given score files and computes EER, HTER.
It also is able to plot CMC and ROC curves."""

# matplotlib stuff
import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec


import facereclib
from .. import utils
import argparse
import bob
import numpy, scipy.stats, math
import os

# enable LaTeX interpreter
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('lines', linewidth = 4)
# increase the default font size
matplotlib.rc('font', size=18)



def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-D', '--directory', help = "If given, files will be read and written to this directory")
  parser.add_argument('-g', '--ground-truth-file', default = "ground_truth.txt", help = "The file containing the ground truth eye locations")
  parser.add_argument('-f', '--landmark-files', nargs='+', default=[], help = "The files containing the detected landmarks")

  parser.add_argument('-l', '--legends', nargs='+', help = "A list of legend strings used for the plots; must be the same length as the --landmark-files.")
  parser.add_argument('-w', '--output', default = 'PTP.pdf', help = "The Point-To-Point error curves will be plotted into the given pdf file.")

#  parser.add_argument('-t', '--title', default='Eye detection errors', help = "The title of the plot")

  parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)

  facereclib.utils.add_logger_command_line_option(parser)

  # parse arguments
  args = parser.parse_args(command_line_parameters)

  if args.legends is None:
    args.legends = args.landmark_files[:]

  if args.directory is not None:
    args.ground_truth_file = os.path.join(args.directory, args.ground_truth_file)
    args.output = os.path.join(args.directory, args.output)
    for i in range(len(args.landmark_files)):
      args.landmark_files[i] = os.path.join(args.directory, args.landmark_files[i])

  facereclib.utils.set_verbosity_level(args.verbose)

  return args


def _read_landmark_file(filename):
  landmarks = {}
  with open(filename) as f:
    for line in f:
      line = line.rstrip()
      if not line or line[0] == '#':
        continue
      splits = line.split()
      landmarks[splits[0]] = [float(splits[i]) for i in range(1,len(splits))]

  return landmarks


def _ptp_all(ground_truth, detected):
  ptps = []
  for key in ground_truth:
    if key not in detected or not detected[key]:
      facereclib.utils.warn("The detected eye locations for file %s are not available" % key)
    else:
      gt = ground_truth[key]
      dt = detected[key]
      inter_eye_distance = math.sqrt((gt[0] - gt[2])**2 + (gt[1] - gt[3])**2)
      errors = [(dt[i] - gt[i]) / inter_eye_distance for i in range(len(gt))]
      ptp = 0.
      for i in range(0, len(ground_truth[key]), 2):
        ptp += math.sqrt((errors[0])**2 + (errors[1])**2)
      ptps.append(ptp / len(ground_truth[key]) * 2.)
  return ptps


def main(command_line_parameters=None):
  """Reads score files, computes error measures and plots curves."""

  args = command_line_arguments(command_line_parameters)
  # read ground truth
  ground_truth = _read_landmark_file(args.ground_truth_file)

  ptps = []
  for landmark_file in args.landmark_files:
    landmarks = _read_landmark_file(landmark_file)
    ptps.append(_ptp_all(ground_truth, landmarks))


  figure = plt.figure(figsize=(10,5))
  for i,p in enumerate(ptps):
    p_hist, first, last, outliers = scipy.stats.cumfreq(p, 100, (0,0.5))
    plt.plot(p_hist / len(p) * 100., label=args.legends[i] + " (+%d)" % outliers)
  plt.xticks(numpy.arange(0, 101, 20), numpy.arange(0, 51, 10))
  plt.xlabel("Point-To-Point error value in \% inter-eye-distance")
  plt.legend(loc=4)

  plt.savefig(args.output)


