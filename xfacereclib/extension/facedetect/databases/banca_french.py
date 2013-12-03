#!/usr/bin/env python

import xbob.db.detection.filelist
import os

base_dir = "/idiap/group/biometric/databases/facedetect/BANCA-french"
# The filelist database interface for the MIT-CMU database.

database = xbob.db.detection.filelist.Database(
  image_directory = os.path.join(base_dir, 'data'),
  image_extensions = (".pgm", ),
  annotation_directory = os.path.join(base_dir, 'annotations'),
  annotation_type = 'named',
  list_base_directory = os.path.join(base_dir, 'filelists'),
  keep_read_lists_in_memory = False
)

