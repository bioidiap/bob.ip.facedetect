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

import facereclib
from .. import utils
import argparse
import bob
import numpy, math
import os

# matplotlib stuff
import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages

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

  parser.add_argument('-d', '--files', required=True, nargs='+', help = "A list of score files to evaluate.")

  parser.add_argument('-s', '--directory', default = '.', help = "A directory, where to find the --dev-files and the --eval-files")

  parser.add_argument('-l', '--legends', nargs='+', help = "A list of legend strings used for ROC, CMC and DET plots; if given, must be the same number than --dev-files.")
  parser.add_argument('-F', '--froc', default = 'FROC.pdf', help = "If given, FROC curves will be plotted into the given pdf file.")

  parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)

  facereclib.utils.add_logger_command_line_option(parser)

  # parse arguments
  args = parser.parse_args(command_line_parameters)

  facereclib.utils.set_verbosity_level(args.verbose)

  if args.legends and len(args.dev_files) != len(args.legends):
    facereclib.utils.error("The number of --dev-files (%d) and --legends (%d) are not identical" % (len(args.dev_files), len(args.legends)))

  # update legends when they are not specified on command line
  if args.legends is None:
    args.legends = args.files


  return args

def _plot_froc(fa, dr, colors, labels, title):
  figure = mpl.figure()
  # plot FAR and FRR for each algorithm
  for i in range(len(fa)):
    mpl.semilogx(fa[i], [100.0*f for f in  dr[i]], color=colors[i], lw=2, ms=10, mew=1.5, label=labels[i])

  # finalize plot
  mpl.xticks((1, 10, 100, 1000, 10000), ('1', '10', '100', '1000', '10000'))
  mpl.xlabel('False Alarm')
  mpl.ylabel('Detection Rate (\%)')
  mpl.grid(True, color=(0.6,0.6,0.6))
  mpl.legend(loc=4)
  mpl.title(title)

  return figure


def read_score_file(filename):
  positives = []
  negatives = []
  ground_truth = 0
  with open(filename) as f:
    while f:
      line = f.readline().rstrip()
      if not len(line):
        break
      face_count = int(line.split()[-1])
      ground_truth += face_count
      # for each face in the image, get the detection scores
      for c in range(face_count):
        splits = f.readline().rstrip().split()
        # here, we only take the first value as detection score
        if len(splits):
          positives.append(float(splits[0]))
      # now, read negative scores
      splits = f.readline().rstrip().split()
      negatives.extend([float(v) for v in splits])

  return (ground_truth, positives, negatives)


def main(command_line_parameters=None):
  """Reads score files, computes error measures and plots curves."""

  args = command_line_arguments(command_line_parameters)

  # get some colors for plotting
  cmap = mpl.cm.get_cmap(name='hsv')
  colors = [cmap(i) for i in numpy.linspace(0, 1.0, len(args.files)+1)]

  # First, read the score files
  facereclib.utils.info("Loading %d score files" % len(args.files))

  scores = [read_score_file(os.path.join(args.directory, f)) for f in args.files]

  false_alarms = []
  detection_rate = []
  facereclib.utils.info("Computing FROC curves")
  for score in scores:
    # compute 20 thresholds
    tmin = min(score[2])
    tmax = max(score[2])
    thresholds = [tmin + x/20. * (tmax - tmin) for x in range(21)]
    false_alarms.append([])
    detection_rate.append([])
    for threshold in thresholds:
      detection_rate[-1].append(numpy.count_nonzero(numpy.array(score[1]) > threshold) / float(score[0]))
      false_alarms[-1].append(numpy.count_nonzero(numpy.array(score[2]) > threshold))

  print (thresholds, false_alarms, detection_rate, colors, args.legends)

  facereclib.utils.info("Plotting FROC curves to file '%s'" % args.froc)
  # create a multi-page PDF for the ROC curve
  pdf = PdfPages(args.froc)
  # create a separate figure for dev and eval
  pdf.savefig(_plot_froc(false_alarms, detection_rate, colors, args.legends, 'FROC'))
  pdf.close()

