import matplotlib.pyplot as plt
from bob.io.base import load
from bob.io.base.test_utils import datafile
from bob.io.image import imshow
from bob.ip.facedetect.tinyface import TinyFacesDetector
from matplotlib.patches import Rectangle

# load colored test image
color_image = load(datafile("test_image_multi_face.png", "bob.ip.facedetect"))
is_mxnet_available = True
try:
    import mxnet
except Exception:
    is_mxnet_available = False

if not is_mxnet_available:
    imshow(color_image)
else:

    # detect all faces
    detector = TinyFacesDetector()
    detections = detector.detect(color_image)

    imshow(color_image)
    plt.axis("off")

    for annotations in detections:
        topleft = annotations["topleft"]
        bottomright = annotations["bottomright"]
        size = bottomright[0] - topleft[0], bottomright[1] - topleft[1]
        # draw bounding boxes
        plt.gca().add_patch(
            Rectangle(
                topleft[::-1],
                size[1],
                size[0],
                edgecolor="b",
                facecolor="none",
                linewidth=2,
            )
        )