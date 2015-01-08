#!/usr/bin/env python

import xbob.db.detection.filelist
import os

base_dir = "/idiap/group/biometric/databases/facedetect/web"

# The filelist database interface for the cinema database.

database = xbob.db.detection.filelist.Database(
  image_directory = "/idiap/group/vision/visidiap/databases/web/images",
  image_extensions = (".png", ),
  annotation_directory = os.path.join(base_dir, 'annotations'),
  annotation_type = 'named',
  list_base_directory = os.path.join(base_dir, 'filelists'),
  keep_read_lists_in_memory = False
)

