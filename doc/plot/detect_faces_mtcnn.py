from bob.io.image import imshow
from bob.io.base import load
from bob.io.base.test_utils import datafile
from bob.ip.facedetect import MTCNNAnnotator
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# load colored test image
color_image = load(
    datafile("test_image_multi_face.png", "bob.ip.facedetect")
)
image_batch = [color_image]

# Detect all faces
detector = MTCNNAnnotator()
detections_batch = detector.transform_multi_face(image_batch, max_face_nb=None)
detections = detections_batch[0]

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
    # draw face landmarks
    for key, color in (
        ("reye", "r"),
        ("leye", "g"),
        ("nose", "b"),
        ("mouthright", "k"),
        ("mouthleft", "w"),
    ):
        plt.gca().add_patch(
            Circle(
                annotations[key][::-1],
                radius=2,
                edgecolor=color,
                facecolor="none",
                linewidth=2,
            )
        )
    # show quality of detections
    plt.text(
        topleft[1], topleft[0], round(annotations["quality"], 3), color="b", fontsize=14
    )
