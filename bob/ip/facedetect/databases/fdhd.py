#!/usr/bin/env python

import xbob.db.detection.filelist
import os

base_dir = "/idiap/group/biometric/databases/facedetect/FDHD-background"

# The filelist database interface for the FDHD background

database = xbob.db.detection.filelist.Database(
  image_directory = os.path.join(base_dir, 'data'),
  image_extensions = (".jpeg", ),
  annotation_directory = None,
  annotation_type = None,
  list_base_directory = os.path.join(base_dir, 'filelists'),
  keep_read_lists_in_memory = False
)

