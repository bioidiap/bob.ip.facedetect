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
  parser.add_argument('-d', '--detected-file', default = "detections.txt", help = "The file containing the expected eye locations based on the face detector")
  parser.add_argument('-g', '--ground-truth-file', default = "ground_truth.txt", help = "The file containing the ground truth eye locations")
  parser.add_argument('-l', '--landmark-file', default = "landmarks.txt", help = "The file containing the detected landmarks")
  parser.add_argument('-f', '--flandmark-file', default = "flandmark.txt", help = "The file containing the landmarks from xbob.flandmark")
  parser.add_argument('-a', '--all-landmarks', action = "store_true", help = "If given, Point-to-Point errors will be computes with all landmarks (not only with the eyes)")

#  parser.add_argument('-l', '--legends', nargs='+', help = "A list of legend strings used for ROC, CMC and DET plots; if given, must be the same number than --files plus --baselines.")
  parser.add_argument('-w', '--output', help = "FROC curves will be plotted into the given pdf file (default: Errors.pdf).")

  parser.add_argument('-t', '--title', default='Eye detection errors', help = "The title of the plot")

  parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)

  facereclib.utils.add_logger_command_line_option(parser)

  # parse arguments
  args = parser.parse_args(command_line_parameters)

  if args.directory is not None:
    args.detected_file = os.path.join(args.directory, args.detected_file)
    args.ground_truth_file = os.path.join(args.directory, args.ground_truth_file)
    if args.landmark_file is not None:
      args.landmark_file = os.path.join(args.directory, args.landmark_file)
    if args.flandmark_file is not None:
      args.flandmark_file = os.path.join(args.directory, args.flandmark_file)
    if args.output is None:
      args.output = os.path.join(args.directory, "Errors_%s.pdf" % os.path.split(args.directory)[-1])

  if args.output is None:
    args.output = "Errors.pdf"
  facereclib.utils.set_verbosity_level(args.verbose)

  return args


def _plot_errors(hist_r, hist_l, count_r, count_l):
  # define sub-figures
  grid = gridspec.GridSpec(1,2)
  grid.update(left=0.09, right=0.8, top=0.97, bottom=0.06, wspace=0.3, hspace=0.4)

  h_max = max(numpy.max(hist_r), numpy.max(hist_l))

  figure = plt.gcf()

  plt.subplot(grid[0])
  plt.imshow(hist_r, extent=[-10, 10, -10, 10], interpolation='nearest')
  plt.clim([0, h_max])
  plt.xticks(range(-10,11,5))
  plt.yticks(range(-10,11,5))
  plt.ylabel("Pixel (@inter-eye-distance 33)")
  plt.title("Right eye, outliers: %d" % count_r)

  plt.subplot(grid[1])
  i = plt.imshow(hist_l, extent=[-10, 10, -10, 10], interpolation='nearest')
  plt.clim([0, h_max])
  plt.xticks(range(-10,11,5))
  plt.yticks(range(-10,11,5))
  plt.title("Left eye, outliers: %d" % count_l)

  cbar_ax = figure.add_axes([0.85, 0.15, 0.05, 0.7])
  figure.colorbar(i, cax=cbar_ax)

def _plot_cumulative(J, P):
  # define sub-figures
  grid = gridspec.GridSpec(2,1)
  grid.update(left=0.09, right=0.96, top=0.9, bottom=0.15, wspace=0.3, hspace=0.4)

  plt.subplot(grid[0])
  for i,j in enumerate(J):
    j_hist, first, last, outliers = scipy.stats.cumfreq(j, 100, (0,0.2))
    plt.plot(j_hist / len(j) * 100., label=["detected", "landmarks", "flandmark"][i] + " (+%d)" % outliers)
  plt.xticks(numpy.arange(0, 101, 50), numpy.arange(0, 0.21, 0.1) )
  plt.xlabel("Jesorsky error value")
  plt.legend(loc=4)

  plt.subplot(grid[1])
  for i,p in enumerate(P):
    p_hist, first, last, outliers = scipy.stats.cumfreq(p, 100, (0,1))
    plt.plot(p_hist / len(p) * 100., label=["detected", "landmarks", "flandmark"][i] + " (+%d)" % outliers)
  plt.xticks(numpy.arange(0, 101, 20), numpy.arange(0, 1.01, 0.2))
  plt.xlabel("Point-To-Point error value")
  plt.legend(loc=4)


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


def _compute_error(ground_truth, detected):

  # relative error
  relative = numpy.zeros(4, numpy.float64)
  error_y = 0.
  error_x = 0.

  jesorsky = []
  ptp = []

  hist_r = numpy.zeros((21,21), numpy.int)
  hist_l = numpy.zeros((21,21), numpy.int)

  outside_r = 0
  outside_l = 0

  # Compute the error between the GT locations and the detected locations
  number_of_detections = 0.
  for key in ground_truth:
    if key not in detected or not detected[key]:
      facereclib.utils.warn("The detected eye locations for file %s are not available" % key)
    else:
      number_of_detections += 1.
      gt = ground_truth[key]
      dt = detected[key]
      # order is: re_y, re_x, le_y, le_x
      inter_eye_distance = math.sqrt((gt[0] - gt[2])**2 + (gt[1] - gt[3])**2)
      errors = [(dt[i] - gt[i]) / inter_eye_distance for i in (0,1,2,3)]

      # compute histogram
      bins = [int(round(e * 33.)) + 10 for e in errors]
      if bins[0] < 0 or bins[0] > 20 or bins[1] < 0 or bins[1] > 20:
        outside_r += 1
      else:
        hist_r[bins[0], bins[1]] += 1
      if bins[2] < 0 or bins[2] > 20 or bins[3] < 0 or bins[3] > 20:
        outside_l += 1
      else:
        hist_l[bins[2], bins[3]] += 1

      # compute Jesorsky measure
      jesorsky.append(max(errors[0]**2 + errors[1]**2, errors[2]**2 + errors[3]**2))
      ptp.append(math.sqrt(errors[0]**2 + errors[1]**2) + math.sqrt(errors[2]**2 + errors[3]**2)/2)

  # plot histograms
  _plot_errors(hist_r, hist_l, outside_r, outside_l)


  # return Jesorsky and Point-To-Point errors
  return jesorsky, ptp

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
  pdf = PdfPages(args.output)

  # read errors
  detected = _read_landmark_file(args.detected_file)

  # read ground truth
  ground_truth = _read_landmark_file(args.ground_truth_file)

  figure = plt.figure(figsize=(10,5))
  jesorsky, ptp = _compute_error(ground_truth, detected)
  figure.suptitle("Ground truth vs. detected")
  pdf.savefig(figure)

  J = [jesorsky]
  P = [ptp]

  # if given, also read the landmark file
  if args.landmark_file and os.path.exists(args.landmark_file):
    landmarks = _read_landmark_file(args.landmark_file)
    lm_eyes = {}
    for key, lm in landmarks.iteritems():
      lm_eyes[key] = [lm[0], lm[1], lm[2], lm[3]]

    figure = plt.figure(figsize=(10,5))
    jesorsky, ptp = _compute_error(ground_truth, lm_eyes)
    if args.all_landmarks:
      # overwrite the PTP value with all
      ptp = _ptp_all(ground_truth, landmarks)
    figure.suptitle("Ground truth vs. landmarks")
    pdf.savefig(figure)

    J.append(jesorsky)
    P.append(ptp)

  # if given, also read the flandmark file
  if args.flandmark_file and os.path.exists(args.flandmark_file):
    landmarks = _read_landmark_file(args.flandmark_file)
    lm_eyes = {}
    for key, lm in landmarks.iteritems():
      lm_eyes[key] = [(lm[2] + lm[10])/2., (lm[3] + lm[11])/2., (lm[4] + lm[12])/2., (lm[5] + lm[13])/2.]

    figure = plt.figure(figsize=(10,5))
    jesorsky, ptp = _compute_error(ground_truth, lm_eyes)
    figure.suptitle("Ground truth vs. flandmark")
    pdf.savefig(figure)

    J.append(jesorsky)
#    if not args.all_landmarks:
    # for FLandmark, we don't have all corresponding landmarks...
    P.append(ptp)

  figure = plt.figure(figsize=(10,5))
  _plot_cumulative(J, P)
  figure.suptitle("Cumulative error distributions")
  pdf.savefig(figure)

  pdf.close()


