
#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <bob/config.h>
#include <bob/python/ndarray.h>
#include <bob/python/gil.h>
#include "features.h"
#include <sstream>

using namespace boost::python;


static boost::shared_ptr<FeatureExtractor> init_from_vector_of_lbp(const blitz::TinyVector<int,2>& patchSize, object extractors){
  stl_input_iterator<bob::ip::LBP> dbegin(extractors), dend;
  std::vector<bob::ip::LBP> lbps(dbegin, dend);
  return boost::make_shared<FeatureExtractor>(patchSize, lbps);
}

static void extract_all(const FeatureExtractor& extractor, const BoundingBox& boundingBox, bob::python::ndarray data, int datasetIndex){
  blitz::Array<uint16_t,2> d = data.bz<uint16_t,2>();
  extractor.extractAll(boundingBox, d, datasetIndex);
}

static void extract_some(const FeatureExtractor& extractor, const BoundingBox& boundingBox, bob::python::ndarray features){
  blitz::Array<uint16_t,1> f = features.bz<uint16_t,1>();
  extractor.extractSome(boundingBox, f);
}

static void extract_indexed(const FeatureExtractor& extractor, const BoundingBox& boundingBox, bob::python::ndarray features, bob::python::const_ndarray indices){
  blitz::Array<uint16_t,1> f = features.bz<uint16_t,1>();
  const blitz::Array<int32_t,1> i = indices.bz<int32_t,1>();
  extractor.extractIndexed(boundingBox, f, i);
}


static void extract_all_p(const FeatureExtractor& extractor, const BoundingBox& boundingBox, bob::python::ndarray data, int datasetIndex){
  blitz::Array<uint16_t,2> d = data.bz<uint16_t,2>();
  bob::python::no_gil t;
  extractor.extractAll(boundingBox, d, datasetIndex);
}

static void extract_some_p(const FeatureExtractor& extractor, const BoundingBox& boundingBox, bob::python::ndarray features){
  blitz::Array<uint16_t,1> f = features.bz<uint16_t,1>();
  bob::python::no_gil t;
  extractor.extractSome(boundingBox, f);
}

static void extract_indexed_p(const FeatureExtractor& extractor, const BoundingBox& boundingBox, bob::python::ndarray features, bob::python::const_ndarray indices){
  blitz::Array<uint16_t,1> f = features.bz<uint16_t,1>();
  const blitz::Array<int32_t,1> i = indices.bz<int32_t,1>();
  bob::python::no_gil t;
  extractor.extractIndexed(boundingBox, f, i);
}

static blitz::TinyVector<double,2> mv_p(const FeatureExtractor& extractor, const BoundingBox& boundingBox){
  bob::python::no_gil t;
  return extractor.meanAndVariance(boundingBox);
}



static object get_extractors(const FeatureExtractor& extractor){
  const std::vector<bob::ip::LBP>& lbps = extractor.getExtractors();
  list ret;
  for (std::vector<bob::ip::LBP>::const_iterator it = lbps.begin(); it != lbps.end(); ++it){
    ret.append(*it);
  }
  return ret;
}

static std::string bb_str(const BoundingBox& bb){
  std::ostringstream s;
  s << "<BB top=" << bb.top() << ", left=" << bb.left() << ", bottom=" << bb.bottom() << ", right=" << bb.right() << ">";
  return s.str();
}

template <typename T>
static void prepare(FeatureExtractor& self, const blitz::Array<T,2>& image, double scale, bool compute_integral_square_image){
  // release gil
  bob::python::no_gil t;
  self.prepare(image, scale, compute_integral_square_image);
}


static object prune_detections(object detections, const blitz::Array<double,1>& predictions, double threshold, const int number_of_detections){
  // extract the bounding boxes
  stl_input_iterator<BoundingBox> dbegin(detections), dend;
  std::vector<BoundingBox> bbs(dbegin, dend);

  blitz::Array<double,1> pruned_predictions;
  std::vector<BoundingBox> pruned_bbs;

  // prune
  pruneDetections(bbs, predictions, threshold, pruned_bbs, pruned_predictions, number_of_detections);

  // convert back to list
  list ret;
  for (std::vector<BoundingBox>::const_iterator it = pruned_bbs.begin(); it != pruned_bbs.end(); ++it){
    ret.append(*it);
  }

  return make_tuple(ret, pruned_predictions);
}

static object overlapping_detections(object detections, const blitz::Array<double,1>& predictions, double threshold){
  // extract the bounding boxes
  stl_input_iterator<BoundingBox> dbegin(detections), dend;
  std::vector<BoundingBox> bbs(dbegin, dend);

  blitz::Array<double,1> overlapping_predictions;
  std::vector<BoundingBox> overlapping_bbs;

  // get overlapping bounding boxes
  bestOverlap(bbs, predictions, threshold, overlapping_bbs, overlapping_predictions);

  // convert back to list
  list ret;
  for (std::vector<BoundingBox>::const_iterator it = overlapping_bbs.begin(); it != overlapping_bbs.end(); ++it){
    ret.append(*it);
  }

  return make_tuple(ret, overlapping_predictions);
}



BOOST_PYTHON_MODULE(_features) {
  bob::python::setup_python("Bindings for the xfacereclib.extension.facedetect machines.");

  class_<BoundingBox, boost::shared_ptr<BoundingBox> >("BoundingBox", "A bounding box class storing top, left, height and width of an rectangle", no_init)
    .def(init<double, double, double, double>((arg("self"), arg("top"), arg("left"), arg("height"), arg("width")), "Constructor taking top-left position and height, width of the rectangle"))
    .def(init<int, int, int, int>((arg("self"), arg("top"), arg("left"), arg("height"), arg("width")), "Constructor taking top-left position and height, width of the rectangle"))
    .def(init<BoundingBox>((arg("self"), arg("other")), "Copy constructor"))
    .def(self == self)

    .def("shift", &BoundingBox::shift, (arg("self"), arg("y"), arg("x")), "Returns a shifted version of this BoundingBox")
    .def("scale", &BoundingBox::scale, (arg("self"), arg("scale")), "Returns a scaled version of this BoundingBox (transformation center is (0,0))")
    .def("scale_centered", &BoundingBox::scaleCentered, (arg("self"), arg("scale")), "Returns a scaled version of this BoundingBox (transformation center is the center of the bounding box)")
    .def("mirror_x", &BoundingBox::mirrorX, (arg("self"), arg("width")), "Returns a copy of this bounding box mirrored for the given image width")
    .def("overlap", &BoundingBox::overlap, (arg("self"), arg("other")), "Returns the overlap between this and the given bounding box.")
    .def("similarity", &BoundingBox::similarity, (arg("self"), arg("other")), "Computes the Jaccard similarity index between this and the given BoundingBox")

    .def("__str__", &bb_str, (arg("self")), "Returns a string representing this bounding box")
    .def("__repr__", &bb_str, (arg("self")), "Returns a string representing this bounding box")

    .add_property("top", &BoundingBox::itop)
    .add_property("left", &BoundingBox::ileft)
    .add_property("height", &BoundingBox::iheight)
    .add_property("width", &BoundingBox::iwidth)
    .add_property("bottom", &BoundingBox::ibottom)
    .add_property("right", &BoundingBox::iright)
    .add_property("area", &BoundingBox::area)

    .add_property("top_f", &BoundingBox::top)
    .add_property("left_f", &BoundingBox::left)
    .add_property("height_f", &BoundingBox::height)
    .add_property("width_f", &BoundingBox::width)
    .add_property("bottom_f", &BoundingBox::bottom)
    .add_property("right_f", &BoundingBox::right)

  ;

  def("prune_detections", &prune_detections, (arg("detections"), arg("predictions"), arg("threshold"), arg("number_of_detections")=-1), "Prunes the given detected bounding boxes according to their predictions and returns the pruned bounding boxes and their predictions");
  def("overlapping_detections", &overlapping_detections, (arg("detections"), arg("predictions"), arg("threshold")), "Returns the detections and predictions that overlap with the best detection.");

  class_<FeatureExtractor, boost::shared_ptr<FeatureExtractor> >("FeatureExtractor",  "A class to extract several kinds of LBP features", no_init)
    .def(init<const blitz::TinyVector<int,2>&, const bob::ip::LBP& , bool, bool>((arg("self"), arg("patch_size"), arg("template"), arg("overlap")=false, arg("square")=false), "Creates LBP extractor classes of the given template for all possible radii/ block size."))
    .def("__init__", make_constructor(&init_from_vector_of_lbp, default_call_policies(), (arg("patch_size"), arg("extractors"))), "Uses the given list of LBP extractors.")
    .def(init<const FeatureExtractor&>((arg("self"), arg("other")), "Copy constructor"))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("file")), "Creates a feature extractor by reading it from file"))
    .def("append", &FeatureExtractor::append, "Appends the given feature extractor to this one.")

    .def("load", &FeatureExtractor::load, (arg("self"), arg("hdf5File")), "Loads the extractors from the given file.")
    .def("save", &FeatureExtractor::save, (arg("self"), arg("hdf5File")), "Writes the extractors to the given file.")

    .def("prepare", &FeatureExtractor::prepare<double>, (arg("self"), arg("image"), arg("scale"), arg("compute_integral_square_image")=false), "Take the given image to perform the next extraction steps for the given scale")
    .def("prepare", &FeatureExtractor::prepare<uint8_t>, (arg("self"), arg("image"), arg("scale"), arg("compute_integral_square_image")=false), "Take the given image to perform the next extraction steps for the given scale")

    .def("prepare_p", &prepare<double>, (arg("self"), arg("image"), arg("scale"), arg("compute_integral_square_image")=false), "Take the given image to perform the next extraction steps for the given scale")
    .def("prepare_p", &prepare<uint8_t>, (arg("self"), arg("image"), arg("scale"), arg("compute_integral_square_image")=false), "Take the given image to perform the next extraction steps for the given scale")

    .def("mean", &FeatureExtractor::mean, (arg("self"), arg("bounding_box")), "Computes the mean in the given bounding box for the previously prepared image (needs integral image)")
    .def("variance", &FeatureExtractor::variance, (arg("self"), arg("bounding_box")), "Computes the variance in the given bounding box for the previously prepared image (needs integral square image enabled)")
    .def("mean_and_variance", &FeatureExtractor::meanAndVariance, (arg("self"), arg("bounding_box")), "Computes the mean and the variance in the given bounding box for the previously prepared image (needs integral square image enabled)")
    .def("mean_and_variance_p", &mv_p, (arg("self"), arg("bounding_box")), "Computes the mean and the variance in the given bounding box for the previously prepared image (needs integral square image enabled)")

    .def("__call__", &extract_all, (arg("self"), arg("bounding_box"), arg("dataset"), arg("dataset_index")), "Extracts all features into the given dataset of (training) features at the given index.")
    .def("__call__", &extract_some, (arg("self"), arg("bounding_box"), arg("feature_vector") ), "Extracts the features only at the required locations (using self.model_indices)")

    .def("extract", &extract_all, (arg("self"), arg("bounding_box"), arg("dataset"), arg("dataset_index")), "Extracts all features into the given dataset of (training) features at the given index.")
    .def("extract_single", &extract_some, (arg("self"), arg("bounding_box"), arg("feature_vector") ), "Extracts the features only at the required locations (using self.model_indices)")
    .def("extract_indexed", &extract_indexed, (arg("self"), arg("bounding_box"), arg("feature_vector"), arg("indices") ), "Extracts the features only at the required locations")

    .def("extract_p", &extract_all_p, (arg("self"), arg("bounding_box"), arg("dataset"), arg("dataset_index")), "Extracts all features into the given dataset of (training) features at the given index.")
    .def("extract_single_p", &extract_some_p, (arg("self"), arg("bounding_box"), arg("feature_vector") ), "Extracts the features only at the required locations (using self.model_indices)")
    .def("extract_indexed_p", &extract_indexed_p, (arg("self"), arg("bounding_box"), arg("feature_vector"), arg("indices") ), "Extracts the features only at the required locations")

    .def("maximum_label", &FeatureExtractor::getMaxLabel, (arg("self")), "Returns the maximum label the feature extractor will extract.")

    .add_property("image", make_function(&FeatureExtractor::getImage, return_value_policy<copy_const_reference>()), "The (prepared) image the next features will be extracted from.")
    .add_property("model_indices", &FeatureExtractor::getModelIndices, &FeatureExtractor::setModelIndices, "The indices at which the features are extracted")
    .add_property("number_of_features", &FeatureExtractor::numberOfFeatures, "The length of the feature vector that will be extracted by this class")
    .add_property("extractors", &get_extractors, "The LBP extractors used by this class.")
  ;
}
