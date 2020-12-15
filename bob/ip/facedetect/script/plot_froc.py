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

"""This script evaluates the given score files and plots an FROC curve."""

from __future__ import print_function

# matplotlib stuff
import matplotlib

matplotlib.use("pdf")  # avoids TkInter threaded start
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages

import argparse
import numpy, math
import os

import bob.ip.facedetect
import bob.core

logger = bob.core.log.setup("bob.ip.facedetect")


def command_line_options(command_line_arguments):
    """Parse the program options"""

    # set up command line parser
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-d",
        "--files",
        required=True,
        nargs="+",
        help="A list of score files to evaluate.",
    )
    parser.add_argument(
        "-b",
        "--baselines",
        default=[],
        nargs="+",
        help="A list of baseline results to add to the plot",
    )

    parser.add_argument(
        "-D", "--directory", default=".", help="A directory, where to find the --files"
    )
    parser.add_argument(
        "-B",
        "--baseline-directory",
        default=".",
        help="A directory, where to find the --baselines",
    )

    parser.add_argument(
        "-R",
        "--auto-baselines",
        choices=("bioid", "mit-cmu"),
        help="Automatically add the baselines for the given database",
    )

    parser.add_argument(
        "-l",
        "--legends",
        nargs="+",
        help="A list of legend strings used for ROC, CMC and DET plots; if given, must be the same number than --files plus --baselines.",
    )
    parser.add_argument(
        "-w",
        "--output",
        default="FROC.pdf",
        help="If given, FROC curves will be plotted into the given pdf file.",
    )
    parser.add_argument(
        "-c",
        "--count-detections",
        action="store_true",
        help="Counts the number of detections (positive is higher than negative, per file).",
    )
    parser.add_argument(
        "-n",
        "--max",
        type=int,
        nargs=2,
        default=(160, 70),
        help="The highest false alarms and the lowest detection rate to plot",
    )
    parser.add_argument("-t", "--title", default="FROC", help="The title of the plot")

    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)

    # add verbosity option
    bob.core.log.add_command_line_option(parser)
    args = parser.parse_args(command_line_arguments)
    bob.core.log.set_verbosity_level(logger, args.verbose)

    if args.legends is not None:
        count = len(args.files) + (
            len(args.baselines) if args.baselines is not None else 0
        )
        if len(args.legends) != count:
            logger.error(
                "The number of --files (%d) plus --baselines (%d) must be the same as --legends (%d)",
                len(args.files),
                len(args.baselines) if args.baselines else 0,
                len(args.legends),
            )
            args.legends = None

    # update legends when they are not specified on command line
    if args.legends is None:
        args.legends = args.files if not args.baselines else args.files + args.baselines
        args.legends = [l.replace("_", "-") for l in args.legends]

    if args.auto_baselines == "bioid":
        args.baselines.extend(
            [
                "baselines/baseline_detection_froba_mct_BIOID",
                "cosmin/BIOID/face.elbp.proj0.var.levels10.roc",
            ]
        )
        args.legends.extend(["Froba", "Cosmin"])
    elif args.auto_baselines == "mit-cmu":
        args.baselines.extend(
            [
                "baselines/baseline_detection_fcboost_MIT+CMU",
                "baselines/baseline_detection_viola_rapid1_MIT+CMU",
                "cosmin/MIT+CMU/face.elbp.proj0.var.levels10.roc",
            ]
        )
        args.legends.extend(["FcBoost", "Viola", "Cosmin"])

    return args


def _plot_froc(fa, dr, colors, labels, title, max_r):
    figure = mpl.figure()
    # plot FAR and FRR for each algorithm
    for i in range(len(fa)):
        mpl.plot(
            fa[i],
            [100.0 * f for f in dr[i]],
            color=colors[i],
            lw=2,
            ms=10,
            mew=1.5,
            label=labels[i],
        )

    # finalize plot
    #  mpl.xticks((1, 10, 100, 1000, 10000), ('1', '10', '100', '1000', '10000'))
    #  mpl.xticks([100*i for i in range(11)])
    #  mpl.xlim((0.1, 100000))
    mpl.xlim((0, max_r[0]))
    mpl.ylim((max_r[1], 100))
    mpl.grid(True, color=(0.6, 0.6, 0.6))
    mpl.legend(loc=4, prop={"size": 16})
    mpl.title(title)

    return figure


def count_detections(filename):
    detections = 0
    faces = 0
    with open(filename) as f:
        while f:
            line = f.readline().rstrip()
            if not len(line):
                break
            if line[0] == "#":
                continue
            face_count = int(line.split()[-1])
            faces += face_count
            # for each face in the image, get the detection scores
            positives = []
            for c in range(face_count):
                splits = f.readline().rstrip().split()
                # here, we only take the first value as detection score
                if len(splits):
                    positives.append(float(splits[0]))
            # now, read negative scores
            splits = f.readline().rstrip().split()
            if len(splits):
                negative = float(splits[0])
                for positive in positives:
                    if positive > negative:
                        detections += 1
            else:
                detections += face_count

    return (detections, faces)


def read_score_file(filename):
    positives = []
    negatives = []
    ground_truth = 0
    with open(filename) as f:
        while f:
            line = f.readline().rstrip()
            if not len(line):
                break
            if line[0] == "#":
                continue
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


def main(command_line_arguments=None):
    """Reads score files, computes error measures and plots curves."""

    # enable LaTeX interpreter
    matplotlib.rc("text", usetex=True)
    matplotlib.rc("font", family="serif")
    matplotlib.rc("lines", linewidth=4)
    # increase the default font size
    matplotlib.rc("font", size=18)

    args = command_line_options(command_line_arguments)

    # get some colors for plotting
    cmap = mpl.cm.get_cmap(name="hsv")
    count = len(args.files) + (len(args.baselines) if args.baselines else 0)
    colors = [cmap(i) for i in numpy.linspace(0, 1.0, count + 1)]

    # First, read the score files
    logger.info("Loading %d score files" % len(args.files))

    scores = [read_score_file(os.path.join(args.directory, f)) for f in args.files]

    false_alarms = []
    detection_rate = []
    logger.info("Computing FROC curves")
    for score in scores:
        # compute some thresholds
        tmin = min(score[2])
        tmax = max(score[2])
        count = 100
        thresholds = [tmin + float(x) / count * (tmax - tmin) for x in range(count + 2)]
        false_alarms.append([])
        detection_rate.append([])
        for threshold in thresholds:
            detection_rate[-1].append(
                numpy.count_nonzero(numpy.array(score[1]) >= threshold)
                / float(score[0])
            )
            false_alarms[-1].append(
                numpy.count_nonzero(numpy.array(score[2]) >= threshold)
            )
        # to display 0 in a semilogx plot, we have to add a little
    #    false_alarms[-1][-1] += 1e-8

    # also read baselines
    if args.baselines is not None:
        for baseline in args.baselines:
            dr = []
            fa = []
            with open(os.path.join(args.baseline_directory, baseline)) as f:
                for line in f:
                    splits = line.rstrip().split()
                    dr.append(float(splits[0]))
                    fa.append(int(splits[1]))
            false_alarms.append(fa)
            detection_rate.append(dr)

    logger.info("Plotting FROC curves to file '%s'", args.output)
    # create a multi-page PDF for the ROC curve
    pdf = PdfPages(args.output)
    figure = _plot_froc(
        false_alarms, detection_rate, colors, args.legends, args.title, args.max
    )
    mpl.xlabel("False Alarm (of %d pruned)" % len(scores[0][2]))
    mpl.ylabel("Detection Rate in \%% (total %d faces)" % scores[0][0])
    pdf.savefig(figure)
    pdf.close()

    if args.count_detections:
        for i, f in enumerate(args.files):
            det, all = count_detections(f)
            print(
                "The number of detected faces for %s is %d out of %d"
                % (args.legends[i], det, all)
            )

