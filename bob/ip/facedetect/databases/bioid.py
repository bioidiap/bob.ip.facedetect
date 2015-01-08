#!/usr/bin/env python

import xbob.db.detection.filelist
import os

base_dir = "/idiap/group/biometric/databases/facedetect/BioID"
bioid_dir = "/idiap/group/biometric/databases/BioID/original"


# The filelist database interface for the CMU database.

database = xbob.db.detection.filelist.Database(
  image_directory = os.path.join(bioid_dir, 'pgm'),
  image_extensions = (".pgm", ),
  annotation_directory = os.path.join(bioid_dir, 'eye'),
  annotation_extension = '.eye',
  annotation_type = 'lr-eyes',
  list_base_directory = os.path.join(base_dir, 'filelists'),
  keep_read_lists_in_memory = False
)

