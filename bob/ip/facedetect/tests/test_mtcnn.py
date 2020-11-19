
from bob.ip.facedetect.tests.utils import is_library_available

import bob.io.image
import bob.io.base
import bob.io.base.test_utils

import numpy


# An image with one face
face_image = bob.io.base.load(
    bob.io.base.test_utils.datafile(
        'testimage.jpg', 'bob.ip.facedetect'
    )
)

# An image with 6 faces
face_image_multiple = bob.io.base.load(
    bob.io.base.test_utils.datafile(
        'test_image_multi_face.png', 'bob.ip.facedetect'
    )
)


def _assert_mtcnn_annotations(annot):
    """
    Verifies that MTCNN returns the correct coordinates for ``testimage``.
    """
    assert len(annot) == 1, f"len: {len(annot)}; {annot}"
    face = annot[0]
    assert [int(x) for x in face['topleft']] == [68, 76], face
    assert [int(x) for x in face['bottomright']] == [344, 274], face
    assert [int(x) for x in face['reye']] == [180, 129], face
    assert [int(x) for x in face['leye']] == [175, 220], face
    assert numpy.allclose(face['quality'], 0.9998974), face

@is_library_available("tensorflow")
def test_mtcnn():
    """MTCNN should annotate one face correctly."""
    from bob.ip.facedetect.mtcnn import MTCNN
    mtcnn_annotator = MTCNN()
    annot = mtcnn_annotator.annotations(face_image)
    _assert_mtcnn_annotations(annot)

@is_library_available("tensorflow")
def test_mtcnn_multiface():
    """MTCNN should find multiple faces in an image."""
    from bob.ip.facedetect.mtcnn import MTCNN
    mtcnn_annotator = MTCNN()
    annot = mtcnn_annotator.annotations(face_image_multiple)
    assert len(annot) == 6
