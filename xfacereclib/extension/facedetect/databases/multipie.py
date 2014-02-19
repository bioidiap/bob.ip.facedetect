#!/usr/bin/env python

import facereclib
import xbob.db.multipie

multipie_image_directory = "/idiap/resource/database/Multi-Pie/data"
multipie_annotation_directory = "/idiap/group/biometric/annotations/multipie"

# here, we only want to have the cameras that are near-frontal and that contain all frontal annotations
cameras = ('04_1', '05_0', '05_1', '14_0', '13_0')

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.multipie.Database(original_directory = multipie_image_directory, original_extension = ".png", annotation_directory = multipie_annotation_directory),
    name = "multipie",
    original_directory = multipie_image_directory,
    original_extension = ".png",
    annotation_directory = multipie_annotation_directory,
    annotation_type = 'multipie',
    protocol = 'P',
    all_files_options = {'cameras' : cameras}
)

