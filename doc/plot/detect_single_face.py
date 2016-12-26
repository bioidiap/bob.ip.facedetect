import bob.io.base
import bob.io.base.test_utils
import bob.io.image
import bob.ip.facedetect
import bob.ip.draw

# load colored test image
color_image = bob.io.base.load(bob.io.base.test_utils.datafile('testimage.jpg', 'bob.ip.facedetect'))

# detect single face
bounding_box, _ = bob.ip.facedetect.detect_single_face(color_image)

# create figure
bob.ip.draw.box(color_image, bounding_box.topleft, bounding_box.size, color=(255, 0, 0))
bob.io.image.imshow(color_image)
