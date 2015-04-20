#ifndef BOB_IP_FACEDETECT_FEATURES_H
#define BOB_IP_FACEDETECT_FEATURES_H

#include <bob.io.base/HDF5File.h>
#include <bob.ip.base/LBP.h>
#include <bob.ip.base/IntegralImage.h>
#include <bob.ip.base/Affine.h>
#include <bob.core/array_convert.h>
#include <boost/shared_ptr.hpp>
#include <limits.h>

namespace bob { namespace ip { namespace facedetect {

class BoundingBox{
  public:
    // default constructor
    BoundingBox(double top, double left, double height, double width) : m_top(top), m_left(left), m_height(height), m_width(width), m_area(width*height) {}
    // copy constructor
    BoundingBox(const BoundingBox& other) : m_top(other.m_top), m_left(other.m_left), m_height(other.m_height), m_width(other.m_width), m_area(m_width*m_height) {}

    // create boundingbox by shifting
    boost::shared_ptr<BoundingBox> shift(double y, double x) const {return boost::shared_ptr<BoundingBox>(new BoundingBox(m_top + y, m_left + x, m_height, m_width));}
    // create boundingbox by scaling
    boost::shared_ptr<BoundingBox> scale(double scale) const {return boost::shared_ptr<BoundingBox>(new BoundingBox(m_top*scale, m_left*scale, m_height*scale, m_width*scale));}
    // create boundingbox by scaling based on the center of the bounding box
    boost::shared_ptr<BoundingBox> scaleCentered(double scale) const {return boost::shared_ptr<BoundingBox>(new BoundingBox(m_top - m_height/2.*(scale-1.), m_left - m_width/2.*(scale-1.), m_height*scale, m_width*scale));}
    // create a bounding box that is mirrored horizontically, adapted to the image width
    boost::shared_ptr<BoundingBox> mirrorX(int width) const {return boost::shared_ptr<BoundingBox>(new BoundingBox(m_top, width - m_width - m_left, m_height, m_width));}

    boost::shared_ptr<BoundingBox> overlap(const BoundingBox& other) const;

    bool operator == (const BoundingBox& other){return top() == other.top() && left() == other.left() && height() == other.height() && width() == other.width();}

    // query functions
    double top() const {return m_top;}
    double bottom() const {return m_top + m_height;}
    double left() const {return m_left;}
    double right() const {return m_left + m_width;}
    double height() const {return m_height;}
    double width() const {return m_width;}

    blitz::TinyVector<double,2> center() const {return blitz::TinyVector<double,2>(m_top + m_height/2, m_left + m_width/2.);}

    int itop() const {return irnd(top());}
    int ibottom() const {return irnd(bottom());}
    int ileft() const {return irnd(left());}
    int iright() const {return irnd(right());}
    int iheight() const {return irnd(height());}
    int iwidth() const {return irnd(width());}

    double area() const{return m_area;}

    // Jaccard similarity between bounding boxes
    double similarity(const BoundingBox& other) const;

    bool isValidFor(blitz::TinyVector<int,2> shape) const {return m_top >= 0 && m_top + m_height < shape[0] && m_left >= 0 && m_left + m_width < shape[1];}

    bool contains(blitz::TinyVector<double,2> point) const {return point[0] >= m_top && point[1] >= m_left && point[0] < bottom() && point[1] < right();}

  private:
    int irnd(double x) const {return (int)round(x);}

    double m_top, m_left, m_height, m_width;
    double m_area;
};

void pruneDetections(const std::vector<boost::shared_ptr<BoundingBox>>& detections, const blitz::Array<double, 1>& predictions, double threshold, std::vector<boost::shared_ptr<BoundingBox>>& pruned_boxes, blitz::Array<double, 1>& pruned_weights, const int number_of_detections);
void bestOverlap(const std::vector<boost::shared_ptr<BoundingBox>>& detections, const blitz::Array<double, 1>& predictions, double threshold, std::vector<boost::shared_ptr<BoundingBox>>& pruned_boxes, blitz::Array<double, 1>& pruned_weights);

class FeatureExtractor{

  public:
    FeatureExtractor(const blitz::TinyVector<int,2>& patchSize);
    // Creates all possible combinations of LBP extractors using the given template
    FeatureExtractor(const blitz::TinyVector<int,2>& patchSize, const bob::ip::base::LBP& templAte, bool overlap = false, bool square = false, int min_size=1, int max_size=INT_MAX, int distance=1);

    // Uses the given LBP extractors only; Please don't mix MB-LBP with regular LBP's
    FeatureExtractor(const blitz::TinyVector<int,2>& patchSize, const std::vector<boost::shared_ptr<bob::ip::base::LBP>>& extractors);

    // copy constructor
    FeatureExtractor(const FeatureExtractor& other);

    // Reads the LBP extractor types from File
    FeatureExtractor(bob::io::base::HDF5File& file);

    // concatenates the given FeatureExtractor to this one
    void append(const FeatureExtractor& other);

    // append the given LBP extractor ONLY at the given offset positions
    void append(const boost::shared_ptr<bob::ip::base::LBP>& lbp, const std::vector<blitz::TinyVector<int32_t,2> >& offsets);

    void load(bob::io::base::HDF5File& file);
    void save(bob::io::base::HDF5File& file) const;
    const std::vector<boost::shared_ptr<bob::ip::base::LBP>>& getExtractors() const {return m_extractors;}

    // Model indices
    void setModelIndices(const blitz::Array<int32_t,1>& indices) {m_modelIndices.resize(indices.shape()); m_modelIndices = indices;}
    blitz::Array<int32_t,1> getModelIndices() const {return m_modelIndices;}

    // feature information
    int numberOfFeatures() const {return m_featureStarts((int)m_extractors.size());}
    uint16_t getMaxLabel() const {return m_extractors[0]->getMaxLabel();}

    template <typename T>
      void prepare(const blitz::Array<T,2>& image, double scale, bool computeIntegralSquareImage);

    // the prepared image
    const blitz::Array<double,2>& getImage() const {return m_image;}

    // Extract the features
    void extractAll(const BoundingBox& boundingBox, blitz::Array<uint16_t,2>& dataset, int datasetIndex) const;

    void extractSome(const BoundingBox& boundingBox, blitz::Array<uint16_t,1>& featureVector) const;

    void extractIndexed(const BoundingBox& boundingBox, blitz::Array<uint16_t,1>& featureVector, const blitz::Array<int32_t,1>& indices) const;

    double mean(const BoundingBox& boundingBox) const;
    double variance(const BoundingBox& boundingBox) const;
    blitz::TinyVector<double,2> meanAndVariance(const BoundingBox& boundingBox) const;

    blitz::TinyVector<int,2> patchSize() const {return m_patchSize;}

    const boost::shared_ptr<bob::ip::base::LBP> extractor(int32_t index) const {return m_extractors[m_lookUpTable(index,0)];}
    blitz::TinyVector<int32_t,2> offset(int32_t index) const {return blitz::TinyVector<int,2>(m_lookUpTable(index,1), m_lookUpTable(index,2));}

  private:

    void init();

    // look up table storing three information: lbp index, offset y, offset x
    blitz::TinyVector<int,2> m_patchSize;
    blitz::Array<int,2> m_lookUpTable;

    std::vector<boost::shared_ptr<bob::ip::base::LBP>> m_extractors;

    blitz::Array<int32_t,1> m_featureStarts;
    blitz::Array<int32_t,1> m_modelIndices;

    blitz::Array<double,2> m_image;
    blitz::Array<double,2> m_integralImage;
    blitz::Array<double,2> m_integralSquareImage;

    mutable std::vector<blitz::Array<uint16_t,2> > m_featureImages;
    bool m_isMultiBlock;
    bool m_hasSingleOffsets;
};

template <typename T>
  inline void FeatureExtractor::prepare(const blitz::Array<T,2>& image, double scale, bool computeIntegralSquareImage){
    // TODO: implement different MB-LBP behaviour here (i.e., scaling the LBP's instead of scaling the image)

    // scale image
    m_image.resize(bob::ip::base::getScaledShape(image.shape(), scale));
    bob::ip::base::scale(image, m_image);
    if (m_isMultiBlock or computeIntegralSquareImage){
      // compute integral image of scaled image
      m_integralImage.resize(m_image.extent(0)+1, m_image.extent(1)+1);
      if (computeIntegralSquareImage){
        m_integralSquareImage.resize(m_integralImage.extent(0), m_integralImage.extent(1));
        bob::ip::base::integral<double>(m_image, m_integralImage, m_integralSquareImage, true);
      } else {
        bob::ip::base::integral<double>(m_image, m_integralImage, true);
      }
    }
  }

} } } // namespaces

#endif // BOB_IP_FACEDETECT_FEATURES_H
