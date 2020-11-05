import bob.io.base
import bob.io.base.test_utils
import bob.io.image
import bob.ip.facedetect

# load colored test image
color_image = bob.io.base.load(bob.io.base.test_utils.datafile('testimage.jpg', 'bob.ip.facedetect'))

# detect single face
bounding_box, _ = bob.ip.facedetect.detect_single_face(color_image)

# create figure
import skimage.draw
s0,s1 = skimage.draw.rectangle_perimeter(bounding_box.topleft, bounding_box.size, shape=( color_image.shape[1], color_image.shape[2] ))
color_image[:, s0,s1] = 255

