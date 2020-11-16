from bob.ip.facedetect.annotator.mtcnn import MTCNNAnnotator

import bob.io.base
import bob.io.base.test_utils
import bob.io.image

import numpy

face_image = bob.io.base.load(
    bob.io.base.test_utils.datafile(
        'testimage.jpg', 'bob.ip.facedetect'
    )
)

def test_mtcnn_annnotator():
    mtcnn_annotator = MTCNNAnnotator()
    annot = mtcnn_annotator([face_image], expected_face_nb=1)[0][0]

    assert [int(x) for x in annot['topleft']] == [68, 76], annot
    assert [int(x) for x in annot['bottomright']] == [344, 274], annot
    assert [int(x) for x in annot['reye']] == [180, 129], annot
    assert [int(x) for x in annot['leye']] == [175, 220], annot
    assert numpy.allclose(annot['quality'], 0.9998974), annot