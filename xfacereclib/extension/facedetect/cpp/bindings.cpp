
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
  blitz::Array<uint16_t,2> a = data.bz<uint16_t,2>();
  extractor.extract_all(boundingBox, a, datasetIndex);
}

static void extract_some(const FeatureExtractor& extractor, const BoundingBox& boundingBox, bob::python::ndarray data){
  blitz::Array<uint16_t,1> a = data.bz<uint16_t,1>();
  extractor.extract_some(boundingBox, a);
}

static void extract_all_p(const FeatureExtractor& extractor, const BoundingBox& boundingBox, bob::python::ndarray data, int datasetIndex){
  blitz::Array<uint16_t,2> a = data.bz<uint16_t,2>();
  bob::python::no_gil t;
  extractor.extract_all(boundingBox, a, datasetIndex);
}

static void extract_some_p(const FeatureExtractor& extractor, const BoundingBox& boundingBox, bob::python::ndarray data){
  blitz::Array<uint16_t,1> a = data.bz<uint16_t,1>();
  bob::python::no_gil t;
  extractor.extract_some(boundingBox, a);
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
static void prepare(FeatureExtractor& self, const blitz::Array<T,2>& image, double scale){
  // release gil
  bob::python::no_gil t;
  self.prepare(image, scale);
}


BOOST_PYTHON_MODULE(_features) {
  bob::python::setup_python("Bindings for the xfacereclib.extension.facedetect machines.");

  class_<BoundingBox, boost::shared_ptr<BoundingBox> >("BoundingBox", "A bounding box class storing top, left, height and width of an rectangle", no_init)
    .def(init<int, int, int, int>((arg("self"), arg("top"), arg("left"), arg("height"), arg("width")), "Constructor taking top-left position and height, width of the rectangle"))
    .def(init<BoundingBox>((arg("self"), arg("other")), "Copy constructor"))

    .def("shift", &BoundingBox::shift, (arg("self"), arg("y"), arg("x")), "Returns a shifted version of this BoundingBox")
    .def("scale", &BoundingBox::scale, (arg("self"), arg("scale")), "Returns a scaled version of this BoundingBox (transformation center is (0,0))")
    .def("overlap", &BoundingBox::overlap, (arg("self"), arg("other")), "Returns the overlap between this and the given bounding box.")
    .def("similarity", &BoundingBox::similarity, (arg("self"), arg("other")), "Computes the Jaccard similarity index between this and the given BoundingBox")

    .def("__str__", &bb_str, (arg("self")), "Returns a string representing this bounding box")

    .add_property("top", &BoundingBox::top)
    .add_property("left", &BoundingBox::left)
    .add_property("height", &BoundingBox::height)
    .add_property("width", &BoundingBox::width)
    .add_property("bottom", &BoundingBox::bottom)
    .add_property("right", &BoundingBox::right)
    .add_property("area", &BoundingBox::area)
  ;

  class_<FeatureExtractor, boost::shared_ptr<FeatureExtractor> >("FeatureExtractor",  "A class to extract several kinds of LBP features", no_init)
    .def(init<const blitz::TinyVector<int,2>&, const bob::ip::LBP& , bool, bool>((arg("self"), arg("patch_size"), arg("template"), arg("overlap")=false, arg("square")=false), "Creates LBP extractor classes of the given template for all possible radii/ block size."))
    .def("__init__", make_constructor(&init_from_vector_of_lbp, default_call_policies(), (arg("patch_size"), arg("extractors"))), "Uses the given list of LBP extractors.")
    .def(init<const FeatureExtractor&>((arg("self"), arg("other")), "Copy constructor"))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("file")), "Creates a feature extractor by reading it from file"))

    .def("load", &FeatureExtractor::load, (arg("self"), arg("hdf5File")), "Loads the extractors from the given file.")
    .def("save", &FeatureExtractor::save, (arg("self"), arg("hdf5File")), "Writes the extractors to the given file.")

    .def("prepare", &FeatureExtractor::prepare<double>, (arg("self"), arg("image"), arg("scale")), "Take the given image to perform the next extraction steps for the given scale")
    .def("prepare", &FeatureExtractor::prepare<uint8_t>, (arg("self"), arg("image"), arg("scale")), "Take the given image to perform the next extraction steps for the given scale")

    .def("prepare_p", &prepare<double>, (arg("self"), arg("image"), arg("scale")), "Take the given image to perform the next extraction steps for the given scale")
    .def("prepare_p", &prepare<uint8_t>, (arg("self"), arg("image"), arg("scale")), "Take the given image to perform the next extraction steps for the given scale")

    .def("__call__", &extract_all, (arg("self"), arg("bounding_box"), arg("dataset"), arg("dataset_index")), "Extracts all features into the given dataset of (training) features at the given index.")
    .def("__call__", &extract_some, (arg("self"), arg("bounding_box"), arg("feature_vector") ), "Extracts the features only at the required locations (see model_indices)")

    .def("extract", &extract_all, (arg("self"), arg("bounding_box"), arg("dataset"), arg("dataset_index")), "Extracts all features into the given dataset of (training) features at the given index.")
    .def("extract_single", &extract_some, (arg("self"), arg("bounding_box"), arg("feature_vector") ), "Extracts the features only at the required locations (see model_indices)")

    .def("extract_p", &extract_all_p, (arg("self"), arg("bounding_box"), arg("dataset"), arg("dataset_index")), "Extracts all features into the given dataset of (training) features at the given index.")
    .def("extract_single_p", &extract_some_p, (arg("self"), arg("bounding_box"), arg("feature_vector") ), "Extracts the features only at the required locations (see model_indices)")

    .def("maximum_label", &FeatureExtractor::getMaxLabel, (arg("self")), "Returns the maximum label the feature extractor will extract.")

    .add_property("model_indices", &FeatureExtractor::getModelIndices, &FeatureExtractor::setModelIndices, "The indices at which the features are extracted")
    .add_property("number_of_features", &FeatureExtractor::numberOfFeatures, "The length of the feature vector that will be extracted by this class")
    .add_property("extractors", &get_extractors, "The LBP extractors used by this class.")
  ;
}
