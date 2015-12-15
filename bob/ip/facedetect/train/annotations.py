
def read_annotation_file(annotation_file, annotation_type):
  """read_annotation_file(annotation_file, annotation_type) -> annotations
  Reads annotations from the given ``annotation_file``.

  The way, how annotations are read depends on the given ``annotation_type``.
  Depending on the type, one or several annotations might be present in the annotation file.
  Currently, these variants are implemented:

  - ``'lr-eyes'``: Only the eye positions are stored, in a single row, like: ``le_x le_y re_x re_y``, comment lines starting with ``'#'`` are ignored.
  - ``'named'``: Each line of the file contains a name and two floats, like ``reye x y``; empty lines separate between sets of annotations.
  - ``'idiap'``: A special 22 point format, where each line contains the index and the locations, like ``1 x y``.
  - ``'fddb'``: a special format for the FDDB database; empty lines separate between sets of annotations

  Finally, a list of ``annotations`` is returned in the format: ``[{name: (y,x)}]``.

  **Parameters:**

  ``annotation_file`` : str
    The file name of the annotation file to read

  ``annotation_type`` : str (see above)
    The style of annotation file, in which the given ``annotation_file`` is

  **Returns:**

  ``annotations`` : [dict]
    A list of annotations read from the given file, grouped by annotated objects (faces).
    Each annotation is generally specified as the two eye coordinates, i.e., ``{'reye' : (rey, rex), 'leye' : (ley, lex)}``, but other types of annotations might occur as well.
  """
  annotations = [{}]
  with open(annotation_file) as f:
    if annotation_type == 'idiap':
      # This is a special format where we have enumerated annotations, and a 'gender'
      for line in f:
        positions = line.rstrip().split()
        if positions:
          if positions[0].isdigit():
            # position field
            assert len(positions) == 3
            id = int(positions[0])
            annotations[-1]['key%d'%id] = (float(positions[2]),float(positions[1]))
          else:
            # another field, we take the first entry as key and the rest as values
            annotations[-1][positions[0]] = positions[1:]
        elif len(annotations[-1]) > 0:
          # empty line; split between annotations
          annotations.append({})
      # finally, we add the eye center coordinates as the center between the eye corners; the annotations 3 and 8 seem to be the pupils...
      for annotation in annotations:
        if 'key1' in annotation and 'key5' in annotation:
          annotation['reye'] = ((annotation['key1'][0] + annotation['key5'][0])/2., (annotation['key1'][1] + annotation['key5'][1])/2.)
        if 'key6' in annotation and 'key10' in annotation:
          annotation['leye'] = ((annotation['key6'][0] + annotation['key10'][0])/2., (annotation['key6'][1] + annotation['key10'][1])/2.)
    elif annotation_type == 'lr-eyes':
      # In this format, the eyes are given in a single row "le_x le_y re_x re_y", possibly with a comment line
      # There is only a single annotation per image
      for line in f:
        if len(line) and line[0] != '#':
          positions = line.rstrip().split()
          annotations[0]['leye'] = (float(positions[1]),float(positions[0]))
          annotations[0]['reye'] = (float(positions[3]),float(positions[2]))
    elif annotation_type == 'named':
      # In this format, each line contains three entries: "keyword x y"
      for line in f:
        positions = line.rstrip().split()
        if positions:
          annotations[-1][positions[0]] = (float(positions[2]),float(positions[1]))
        elif len(annotations[-1]) > 0:
          # empty line; split between annotations
          annotations.append({})
    elif annotation_type == 'fddb':
      # This is a special format for the FDDB database
      for line in f:
        positions = line.rstrip().split()
        if not len(positions):
          if len(annotations[-1]) > 0:
            # empty line; split between annotations
            annotations.append({})
        elif len(positions) == 2:
          annotations[-1][positions[0]] = float(positions[1])
        elif len(positions) == 3:
          annotations[-1][positions[0]] = (float(positions[2]),float(positions[1]))
        else:
          raise ValueError("Could not interpret line %s of the annotation file" % line)
    else:
      raise ValueError("The given annotation type %s is not known" % annotation_type)
  if not annotations[-1]:
    del annotations[-1]

  return annotations
