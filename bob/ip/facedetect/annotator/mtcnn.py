# Example taken from:
# https://github.com/blaueck/tf-mtcnn/blob/master/mtcnn_tfv2.py


import tensorflow as tf
import pkg_resources
import bob.ip.color
import bob.io.image

from bob.bio.face.annotator import Base

import logging
logger = logging.getLogger(__name__)

class MTCNNAnnotator(Base):

    """MTCNN v1 wrapper for Tensorflow 2. See
    https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html for
    more details on MTCNN.

    Attributes
    ----------
    factor : float
        Factor is a trade-off between performance and speed.
    min_size : int
        Minimum face size to be detected.
    thresholds : list
        Thresholds are a trade-off between false positives and missed detections.
    """
    def __init__(self, min_size=40, factor=0.709, thresholds=[0.6, 0.7, 0.7], **kwargs):
        super(MTCNNAnnotator, self).__init__(**kwargs)
        self._min_size = min_size
        self._factor = factor
        self._thresholds = thresholds
        self._graph_path = pkg_resources.resource_filename(
            __name__, "models/mtcnn.pb"
        )

        # wrap graph function as a callable function
        self._mtcnn_fun = tf.compat.v1.wrap_function(self._mtcnn_fun, [
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        ])

    def _mtcnn_fun(self, img):
        with open(self._graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef.FromString(f.read())

        with tf.device('/cpu:0'):
            prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
                input_map={
                    'input:0': img,
                    'min_size:0': tf.convert_to_tensor(self._min_size, dtype=float),
                    'thresholds:0': tf.convert_to_tensor(self._thresholds, dtype=float),
                    'factor:0': tf.convert_to_tensor(self._factor, dtype=float)
                },
                return_elements=[
                    'prob:0',
                    'landmarks:0',
                    'box:0'],
                name='')
        return box, prob, landmarks

    def detect(self, img):
        """Detects all faces in the image.

        Parameters
        ----------
        img : numpy.ndarray
            An RGB image in matplotlib BGR format.

        Returns
        -------
        tuple
            A tuple of boxes, probabilities, and landmarks.
        """
        box, prob, landmarks = self._mtcnn_fun(img)
        return box.numpy(), prob.numpy(), landmarks.numpy()


    def annotations(self, image_batch):
        """Detects all faces in the image and returns annotations in bob format.

        Parameters
        ----------
        image_batch : numpy.ndarray
            A batch of RGB images in Bob format.

        Returns
        -------
        list
            A list of annotations. Annotations are dictionaries that contain the
            following keys: ``topleft``, ``bottomright``, ``reye``, ``leye``, ``nose``,
            ``mouthright``, ``mouthleft``, and ``quality``.
        """
        all_annotations = []
        for img in image_batch:
            if len(img.shape) == 2:
                img = bob.ip.color.gray_to_rgb(img)

            # assuming img is Bob format and RGB
            assert img.shape[0] == 3, img.shape
            # network expects BGR opencv format
            img = bob.io.image.to_matplotlib(img)
            img = img[..., ::-1]

            boxes, probs, landmarks = self._mtcnn_fun(img)
            boxes, probs, landmarks = boxes.numpy(), probs.numpy(), landmarks.numpy()

            annots = []
            for box, prob, lm in zip(boxes, probs, landmarks):
                topleft = float(box[0]), float(box[1])
                bottomright = float(box[2]), float(box[3])
                right_eye = float(lm[0]), float(lm[5])
                left_eye = float(lm[1]), float(lm[6])
                nose = float(lm[2]), float(lm[7])
                mouthright = float(lm[3]), float(lm[8])
                mouthleft = float(lm[4]), float(lm[9])
                annots.append(
                    {
                        "topleft": topleft,
                        "bottomright": bottomright,
                        "reye": right_eye,
                        "leye": left_eye,
                        "nose": nose,
                        "mouthright": mouthright,
                        "mouthleft": mouthleft,
                        "quality": float(prob),
                    }
                )
            all_annotations.append(annots)
        return all_annotations

    def transform(self, image_batch, expected_face_nb=1, **kwargs):
        """
        Returns the annotations of faces in each images
        """
        all_annotations = self.annotations(image_batch)
        for annots in all_annotations:
            if len(annots) > expected_face_nb:
                annots = annots[:expected_face_nb]
            elif len(annots) < expected_face_nb:
                logger.warning("MTCNN failed to detect a face.")
        return all_annotations