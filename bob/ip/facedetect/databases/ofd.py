#!/usr/bin/env python

import xbob.db.detection.filelist
import os

base_dir = "/idiap/group/biometric/databases/facedetect/onlinefacedetector"

# The filelist database interface for the CMU database.

database = xbob.db.detection.filelist.Database(
  image_directory = "/idiap/group/biometric/databases/onlinefacedetector",
  image_extensions = ('.jpg',),
  annotation_directory = os.path.join(base_dir, 'annotations'),
  annotation_type = 'idiap',
  list_base_directory = os.path.join(base_dir, 'filelists'),
  keep_read_lists_in_memory = False
)

