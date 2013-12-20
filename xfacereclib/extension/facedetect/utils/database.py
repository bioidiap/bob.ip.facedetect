import facereclib


def _annotations(db, file):
  # returns the annotations for the given file object
  import xbob.db.detection.utils
  import xbob.db.verification.utils
  if isinstance(db, xbob.db.detection.utils.Database):
    # detection databases have multiple annotations per file
    return db.annotations(file.id)
  elif isinstance(db, xbob.db.verification.utils.Database):
    # verification databases have just one annotation per file
    return [db.annotations(file.id)]
  elif isinstance(db, facereclib.databases.DatabaseXBob):
    # verification databases have just one annotation per file
    return [db.annotations(file)]
  else:
    raise NotImplementedError("The given database is of no known type.")


def training_image_annot(databases, limit):
  facereclib.utils.info("Collecting training data")
  # open database to collect training images
  training_files = []
  for database in databases:
    facereclib.utils.info("Processing database '%s'" % database)
    db = facereclib.utils.resources.load_resource(database, 'database')
    if isinstance(db, facereclib.databases.DatabaseXBob):
      # collect image name and annotations
      training_files.extend([(db.m_database.original_file_name(f), _annotations(db,f)) for f in db.training_files()])
    else:
      # collect image name and annotations
      training_files.extend([(db.original_file_name(f), _annotations(db,f)) for f in db.training_files()])


  training_files = [training_files[t] for t in facereclib.utils.quasi_random_indices(len(training_files), limit)]

  for file, annot in training_files:
    facereclib.utils.debug("For training file '%s' loaded annotations '%s'" % (file, str(annot)))

  return training_files


def test_image_annot(databases, protocols, limit):
  # open database to collect training images
  test_files = []
  for database, protocol in zip(databases, protocols):
    db = facereclib.utils.resources.load_resource(database, 'database')
    if isinstance(db, facereclib.databases.DatabaseXBob):
      db = db.m_database

    orig_files = db.test_files(protocol=protocol)
    orig_files = [orig_files[t] for t in facereclib.utils.quasi_random_indices(len(orig_files), limit)]
    # collect image name and annotations
    test_files.extend([(db.original_file_name(f), _annotations(db,f), f) for f in orig_files])

  test_files = [test_files[t] for t in facereclib.utils.quasi_random_indices(len(test_files), limit)]

  for _, annot, file in test_files:
    facereclib.utils.debug("For test file '%s' loaded annotations '%s'" % (file.path, str(annot)))

  return test_files



