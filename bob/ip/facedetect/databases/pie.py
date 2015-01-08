#!/usr/bin/env python

import xbob.db.detection.filelist
import os

base_dir = "/idiap/group/biometric/databases/facedetect/CMU-PIE"
image_dir = "/idiap/resource/database/pie_db"

# The filelist database interface for the CMU-PIE database.

database = xbob.db.detection.filelist.Database(
  image_directory = image_dir,
  image_extensions = (".ppm", ),
  annotation_directory = os.path.join(base_dir, 'annotations'),
  annotation_type = 'named',
  list_base_directory = os.path.join(base_dir, 'filelists'),
  keep_read_lists_in_memory = False
)

