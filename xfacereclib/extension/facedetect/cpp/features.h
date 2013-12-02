#include <bob/io/HDF5File.h>
#include <bob/ip/LBP.h>
#include <bob/ip/integral.h>
#include <bob/ip/scale.h>
#include <bob/core/array_convert.h>
#include <boost/shared_ptr.hpp>

class BoundingBox{
  public:
    // default constructor
    BoundingBox(int top, int left, int height, int width) : m_top(top), m_left(left), m_height(height), m_width(width), m_area(width*height) {}
    // copy constructor
    BoundingBox(const BoundingBox& other) : m_top(other.m_top), m_left(other.m_left), m_height(other.m_height), m_width(other.m_width), m_area(m_width*m_height) {}

    // create boundingbox by shifting
    BoundingBox shift(int y, int x) const {return BoundingBox(m_top + y, m_left + x, m_height, m_width);}
    // create boundingbox by scaling
    BoundingBox scale(double scale) const {return BoundingBox(irnd(m_top*scale), irnd(m_left*scale), irnd(m_height*scale), irnd(m_width*scale));}
    // create a bounding box that is mirrored horizontically, adapted to the image width
    BoundingBox mirrorX(int width) const {return BoundingBox(m_top, width - m_width - m_left, m_height, m_width);}

    BoundingBox overlap(const BoundingBox& other) const;

    bool operator == (const BoundingBox& other){return top() == other.top() && left() == other.left() && height() == other.height() && width() == other.width();}

    // query functions
    int top() const {return m_top;}
    int bottom() const {return m_top + m_height - 1;}
    int left() const {return m_left;}
    int right() const {return m_left + m_width - 1;}
    int height() const {return m_height;}
    int width() const {return m_width;}
    int area() const{return m_area;}

    // Jesorsky distance between bounding boxes
    double similarity(const BoundingBox& other) const;

  private:
    int irnd(double x) const {return (int)round(x);}

    int m_top, m_left, m_height, m_width;
    int m_area;
};

void pruneDetections(const std::vector<BoundingBox>& detections, const blitz::Array<double, 1>& predictions, double threshold, std::vector<BoundingBox>& pruned_boxes, blitz::Array<double, 1>& pruned_weights);

class FeatureExtractor{

  public:
    // Creates all possible combinations of LBP extractors using the given template
    FeatureExtractor(const blitz::TinyVector<int,2>& patchSize, const bob::ip::LBP& templAte, bool overlap = false, bool square = false);

    // Uses the given LBP extractors only; Please don't mix MB-LBP with regular LBP's
    FeatureExtractor(const blitz::TinyVector<int,2>& patchSize, const std::vector<bob::ip::LBP>& extractors);

    // copy constructor
    FeatureExtractor(const FeatureExtractor& other);

    // Reads the LBP extractor types from File
    FeatureExtractor(bob::io::HDF5File& file);

    void load(bob::io::HDF5File& file);
    void save(bob::io::HDF5File& file) const;
    const std::vector<bob::ip::LBP>& getExtractors() const {return m_extractors;}

    // Model indices
    void setModelIndices(const blitz::Array<int32_t,1>& indices) {m_modelIndices.resize(indices.shape()); m_modelIndices = indices;}
    blitz::Array<int32_t,1> getModelIndices() const {return m_modelIndices;}

    // feature information
    int numberOfFeatures() const {return m_featureStarts((int)m_extractors.size());}
    uint16_t getMaxLabel() const {return m_extractors[0].getMaxLabel();}

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

  private:

    void init();

    // look up table storing three information: lbp index, offset y, offset x
    blitz::TinyVector<int,2> m_patchSize;
    blitz::Array<int,2> m_lookUpTable;

    std::vector<bob::ip::LBP> m_extractors;

    blitz::Array<int32_t,1> m_featureStarts;
    blitz::Array<int32_t,1> m_modelIndices;

    blitz::Array<double,2> m_image;
    blitz::Array<double,2> m_integralImage;
    blitz::Array<double,2> m_integralSquareImage;

    mutable std::vector<blitz::Array<uint16_t,2> > m_featureImages;
    bool m_isMultiBlock;
};

template <typename T>
  inline void FeatureExtractor::prepare(const blitz::Array<T,2>& image, double scale, bool computeIntegralSquareImage){
    // TODO: implement different MB-LBP behaviour here (i.e., scaling the LBP's instead of scaling the image)

    // scale image
    m_image.resize(bob::ip::getScaledShape<T>(image, scale));
    bob::ip::scale(image, m_image);
    if (m_isMultiBlock or computeIntegralSquareImage){
      // compute integral image of scaled image
      m_integralImage.resize(m_image.extent(0)+1, m_image.extent(1)+1);
      if (computeIntegralSquareImage){
        m_integralSquareImage.resize(m_integralImage.extent(0), m_integralImage.extent(1));
        bob::ip::integral<double>(m_image, m_integralImage, m_integralSquareImage, true);
      } else {
        bob::ip::integral<double>(m_image, m_integralImage, true);
      }
    }
  }

