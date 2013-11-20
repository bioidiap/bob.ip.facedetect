#include <bob/io/HDF5File.h>
#include <bob/ip/LBP.h>
#include <bob/ip/integral.h>
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

    BoundingBox overlap(const BoundingBox& other) const;

    // query functions
    int top() const {return m_top;}
    int bottom() const {return m_top + m_height - 1;}
    int left() const {return m_left;}
    int right() const {return m_left + m_width - 1;}
    int height() const {return m_height;}
    int width() const {return m_width;}
    int area() const{return m_area;}

    // Jesorsky distance between boundingh boxes
    double similarity(const BoundingBox& other) const;

  private:
    int irnd(double x) const {return (int)round(x);}

    int m_top, m_left, m_height, m_width;
    int m_area;
};

inline BoundingBox BoundingBox::overlap(const BoundingBox& other) const{
  // compute intersection rectangle
  int t = std::max(top(), other.top()),
      b = std::min(bottom(), other.bottom()),
      l = std::max(left(), other.left()),
      r = std::min(right(), other.right());
  return BoundingBox(t, l, b-t+1, r-l+1);
}

inline double BoundingBox::similarity(const BoundingBox& other) const{
  // compute intersection rectangle
  double t = std::max(top(), other.top()),
         b = std::min(bottom(), other.bottom()),
         l = std::max(left(), other.left()),
         r = std::min(right(), other.right());

  // no overlap?
  if (l > r || t > b) return 0.;

  // compute overlap
  double intersection = (b-t+1) * (r-l+1);
  return intersection / (area() + other.area() - intersection);
}


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
    void setModelIndices(const blitz::Array<int64_t,1>& indices) {m_modelIndices.resize(indices.shape()); m_modelIndices = indices;}
    blitz::Array<int64_t,1> getModelIndices() const {return m_modelIndices;}

    // feature information
    int numberOfFeatures() const {return m_featureStarts((int)m_extractors.size());}
    uint16_t getMaxLabel() const {return m_extractors[0].getMaxLabel();}

    template <typename T>
      void prepare(const blitz::Array<T,2>& image);

    // Extract the features
    void extract_all(const BoundingBox& boundingBox, blitz::Array<uint16_t,2>& dataset, int datasetIndex) const;

    void extract_some(const BoundingBox& boundingBox, blitz::Array<uint16_t,1>& featureVector) const;


  private:

    void init();

    // look up table storing three information: lbp index, offset y, offset x
    blitz::TinyVector<int,2> m_patchSize;
    blitz::Array<uint8_t,2> m_lookUpTable;

    std::vector<bob::ip::LBP> m_extractors;

    blitz::Array<int64_t,1> m_featureStarts;
    blitz::Array<int64_t,1> m_modelIndices;

    blitz::Array<double,2> m_image;
    blitz::Array<double,2> m_integralImage;

    mutable std::vector<blitz::Array<uint16_t,2> > m_featureImages;
    bool m_isMultiBlock;
};

template <typename T>
  inline void FeatureExtractor::prepare(const blitz::Array<T,2>& image){
    if (m_isMultiBlock){
      m_integralImage.resize(image.extent(0)+1, image.extent(1)+1);
      bob::ip::integral<T>(image, m_integralImage, true);
    } else {
      m_image.resize(image.shape());
      m_image = bob::core::array::cast<double>(image);
    }
  }

