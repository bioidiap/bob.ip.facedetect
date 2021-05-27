from bob.ip.facedetect.tests.utils import is_library_available

import bob.io.image
import bob.io.base
import bob.io.base.test_utils

import numpy


# An image with one face
face_image = bob.io.base.load(
    bob.io.base.test_utils.datafile("testimage.jpg", "bob.ip.facedetect")
)

# An image with 6 faces
face_image_multiple = bob.io.base.load(
    bob.io.base.test_utils.datafile("test_image_multi_face.png", "bob.ip.facedetect")
)


def _assert_tinyface_annotations(annot):
    """
    Verifies that TinyFace returns the correct coordinates for ``testimage``.
    """
    assert len(annot) == 1, f"len: {len(annot)}; {annot}"
    face = annot[0]
    assert [int(x) for x in face['topleft']] == [59, 57], face
    assert [int(x) for x in face['bottomright']] == [338, 284], face
    assert [int(x) for x in face['reye']] == [162, 125], face
    assert [int(x) for x in face['leye']] == [162, 216], face

@is_library_available("mxnet")
def test_tinyface():
    """TinyFace should annotate one face correctly."""
    from bob.ip.facedetect.tinyface import TinyFacesDetector

    tinyface_annotator = TinyFacesDetector()
    annot = tinyface_annotator.detect(face_image)
    _assert_tinyface_annotations(annot)

@is_library_available("mxnet")
def test_tinyface_multiface():
    """TinyFace should find multiple faces in an image."""
    from bob.ip.facedetect.tinyface import TinyFacesDetector
    
    tinyface_annotator = TinyFacesDetector()
    annot = tinyface_annotator.detect(face_image_multiple)
    assert len(annot) == 6
