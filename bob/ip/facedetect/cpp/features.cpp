#include "features.h"
#include <boost/format.hpp>

bob::ip::facedetect::FeatureExtractor::FeatureExtractor(const blitz::TinyVector<int,2>& patchSize)
: m_patchSize(patchSize),
  m_lookUpTable(0,3),
  m_extractors(),
  m_featureStarts(1),
  m_isMultiBlock(false),
  m_hasSingleOffsets(false)
{
  // first feature extractor always starts at zero
  m_featureStarts(0) = 0;
}

bob::ip::facedetect::FeatureExtractor::FeatureExtractor(const blitz::TinyVector<int,2>& patchSize, const bob::ip::base::LBP& templAte, bool overlap, bool square, int min_size, int max_size, int distance)
: m_patchSize(patchSize),
  m_lookUpTable(0,3),
  m_extractors(),
  m_isMultiBlock(templAte.isMultiBlockLBP()),
  m_hasSingleOffsets(false)
{
  // initialize the extractors
  if (!m_isMultiBlock){
    int max_y = std::min(max_size, patchSize[0] / 2);
    int max_x = std::min(max_size, patchSize[1] / 2);
    for (int dy = min_size; dy < max_y; ++dy)
      for (int dx = min_size; dx < max_x / 2; ++dx)
        if (!square || dy == dx){
          // copy setup from template LBP
          boost::shared_ptr<bob::ip::base::LBP> lbp(new bob::ip::base::LBP(templAte));
          // adapt radius / block size
          lbp->setRadii(blitz::TinyVector<double,2>(dy, dx));
          // add LBP
          m_extractors.push_back(lbp);
    }
  } else {
    if (overlap){
      int max_y = std::min(max_size, patchSize[0] - 2);
      int max_x = std::min(max_size, patchSize[1] - 2);
      for (int dy = min_size; dy <= max_y; ++dy)
        for (int dx = min_size; dx <= max_x; ++dx)
          if (!square || dy == dx){
            // copy setup from template LBP
            boost::shared_ptr<bob::ip::base::LBP> lbp(new bob::ip::base::LBP(templAte));
            // adapt radius / block size
            lbp->setBlockSizeAndOverlap(blitz::TinyVector<int,2>(dy, dx), blitz::TinyVector<int,2>(dy-1, dx-1));
            // add LBP
            m_extractors.push_back(lbp);
      }
    } else {
      int max_y = std::min(max_size, patchSize[0] / 3);
      int max_x = std::min(max_size, patchSize[1] / 3);
      for (int dy = min_size; dy <= max_y; ++dy)
        for (int dx = min_size; dx <= max_x; ++dx)
          if (!square || dy == dx){
            // copy setup from template LBP
            boost::shared_ptr<bob::ip::base::LBP> lbp(new bob::ip::base::LBP(templAte));
            // adapt radius / block size
            lbp->setBlockSize(blitz::TinyVector<int,2>(dy, dx));
            // add LBP
            m_extractors.push_back(lbp);
      }
    }
  }

  // compute LookUpTable
  init();
}


bob::ip::facedetect::FeatureExtractor::FeatureExtractor(const blitz::TinyVector<int,2>& patchSize, const std::vector<boost::shared_ptr<bob::ip::base::LBP>>& extractors)
: m_patchSize(patchSize),
  m_lookUpTable(0,3),
  m_extractors(extractors),
  m_hasSingleOffsets(false)
{
  m_isMultiBlock = extractors[0]->isMultiBlockLBP();
  // check if all other lbp extractors have the same multi-block characteristics
  for (auto it = extractors.begin(); it != extractors.end(); ++it){
    if ((*it)->isMultiBlockLBP() != m_isMultiBlock){
      throw std::runtime_error("All LBP variants need to be multiblock or all are not -- a mix is not possible!");
    }
  }

  init();
}

bob::ip::facedetect::FeatureExtractor::FeatureExtractor(const FeatureExtractor& other)
: m_patchSize(other.m_patchSize),
  m_lookUpTable(other.m_lookUpTable),
  m_extractors(other.m_extractors),
  m_featureStarts(other.m_featureStarts),
  m_modelIndices(other.m_modelIndices),
  m_isMultiBlock(other.m_isMultiBlock),
  m_hasSingleOffsets(other.m_hasSingleOffsets)
{
  // we copy everything, except for the internally allocated memory
  m_featureImages.clear();
  if (! m_hasSingleOffsets){
    for (int e = 0; e < (int)m_extractors.size(); ++e){
      blitz::TinyVector<int,2> shape = m_extractors[e]->getLBPShape(m_patchSize, false);
      m_featureImages.push_back(blitz::Array<uint16_t, 2>(shape));
    }
  }
}


bob::ip::facedetect::FeatureExtractor::FeatureExtractor(bob::io::base::HDF5File& file){
  // read information from file
  load(file);
}

void bob::ip::facedetect::FeatureExtractor::append(const FeatureExtractor& other){
  // read information from file
  if (other.m_isMultiBlock != m_isMultiBlock)
    throw std::runtime_error("Cannot append given extractor since multi-block types differ.");

  if (other.m_patchSize[0] != m_patchSize[0] || other.m_patchSize[1] != m_patchSize[1] )
    throw std::runtime_error("Cannot append given extractor since patch sizes differ.");

  // copy LBP classes
  m_extractors.insert(m_extractors.end(), other.m_extractors.begin(), other.m_extractors.end());
  if (other.m_hasSingleOffsets){
    m_hasSingleOffsets = true;
    throw std::runtime_error("This implementation is wrong. When you want to use this functionality, correct it first.");
    m_lookUpTable.reference(other.m_lookUpTable.copy());
    m_featureStarts.reference(other.m_featureStarts.copy());
    m_featureImages.clear();
    for (auto it = other.m_featureImages.begin(); it != other.m_featureImages.end(); ++it){
      m_featureImages.push_back(it->copy());
    }
  } else {
    // re-initialize
    init();
  }
}


void bob::ip::facedetect::FeatureExtractor::append(const boost::shared_ptr<bob::ip::base::LBP>& lbp, const std::vector<blitz::TinyVector<int32_t, 2> >& offsets){
  // read information from file
  if (lbp->isMultiBlockLBP() != m_isMultiBlock && ! m_extractors.empty())
    throw std::runtime_error("Cannot append given extractor since multi-block types differ.");
  m_isMultiBlock = lbp->isMultiBlockLBP();
  m_hasSingleOffsets = true;
  // copy LBP classes
  int lbp_index = m_extractors.size();
  m_extractors.push_back(lbp);
  int oldFeatures = m_featureStarts(m_featureStarts.extent(0)-1);
  int newFeatures = oldFeatures + offsets.size();
  m_featureStarts.resizeAndPreserve(m_featureStarts.extent(0)+1);
  m_featureStarts(m_featureStarts.extent(0)-1) = newFeatures;

  // REMOVE patch images since they are not required!
  m_featureImages.clear();

  // add offsets
  m_lookUpTable.resizeAndPreserve(newFeatures, 3);
  int i = oldFeatures;
  for (auto it = offsets.begin(); it != offsets.end(); ++it, ++i){
    m_lookUpTable(i,0) = lbp_index;
    m_lookUpTable(i,1) = (*it)[0];
    m_lookUpTable(i,2) = (*it)[1];
//    std::cout << i << ":\t" << m_lookUpTable(i,0) << "\t" <<  m_lookUpTable(i,1) << "\t" << m_lookUpTable(i,2) << std::endl;
  }
}


void bob::ip::facedetect::FeatureExtractor::init(){
  // initialize the indices for the full feature vector extraction
  m_featureStarts.resize(m_extractors.size()+1);
  m_featureStarts(0) = 0;
  m_featureImages.clear();
  for (int e = 0; e < (int)m_extractors.size(); ++e){
    blitz::TinyVector<int,2> shape = m_extractors[e]->getLBPShape(m_patchSize, false);
    m_featureStarts(e+1) = m_featureStarts(e) + shape[0] * shape[1];
    m_featureImages.push_back(blitz::Array<uint16_t, 2>(shape));
  }

  // initialize the look up table for indexed feature extraction
  int lutSize = m_featureStarts((int)m_extractors.size());
  m_lookUpTable.resize(lutSize, 3);
  for (int e = 0, index = 0; e < (int)m_extractors.size(); ++e){
    blitz::TinyVector<int,2> shape = m_featureImages[e].shape();
    for (int y = 0; y < shape[0]; ++y)
      for (int x = 0; x < shape[1]; ++x, ++index){
        // index into the extractor
        m_lookUpTable(index,0) = e;
        // offset in y in this extractor
        m_lookUpTable(index,1) = y + m_extractors[e]->getOffset()[0];
        // offset in x in this extractor
        m_lookUpTable(index,2) = x + m_extractors[e]->getOffset()[1];
    }
  }
}

double bob::ip::facedetect::FeatureExtractor::mean(const BoundingBox& boundingBox) const{
  int t = boundingBox.itop(), b = boundingBox.ibottom()-1, l = boundingBox.ileft(), r = boundingBox.iright()-1;
  // compute the mean using the integral image
  double sum = m_integralImage(t, l)
             + m_integralImage(b, r)
             - m_integralImage(t, r)
             - m_integralImage(b, l);

  double pixelCount = boundingBox.area();

  return sum / pixelCount;
}


double bob::ip::facedetect::FeatureExtractor::variance(const BoundingBox& boundingBox) const{
  int t = boundingBox.itop(), b = boundingBox.ibottom()-1, l = boundingBox.ileft(), r = boundingBox.iright()-1;
  // compute the variance using the integral image and the integral square image
  double square = m_integralSquareImage(t, l)
                + m_integralSquareImage(b, r)
                - m_integralSquareImage(t, r)
                - m_integralSquareImage(b, l);

  double sum = m_integralImage(t, l)
             + m_integralImage(b, r)
             - m_integralImage(t, r)
             - m_integralImage(b, l);

  double pixelCount = boundingBox.area();

  return (square - sum*sum/pixelCount) / (pixelCount-1);
}


blitz::TinyVector<double,2> bob::ip::facedetect::FeatureExtractor::meanAndVariance(const BoundingBox& boundingBox) const{
  int t = boundingBox.itop(), b = boundingBox.ibottom()-1, l = boundingBox.ileft(), r = boundingBox.iright()-1;
  // compute the variance using the integral image and the integral square image
  double square = m_integralSquareImage(t, l)
                + m_integralSquareImage(b, r)
                - m_integralSquareImage(t, r)
                - m_integralSquareImage(b, l);

  double sum = m_integralImage(t, l)
             + m_integralImage(b, r)
             - m_integralImage(t, r)
             - m_integralImage(b, l);

  double pixelCount = boundingBox.area();

  return blitz::TinyVector<double,2>(sum / pixelCount, (square - sum*sum/pixelCount) / (pixelCount-1));
}



void bob::ip::facedetect::FeatureExtractor::extractAll(const BoundingBox& boundingBox, blitz::Array<uint16_t,2>& dataset, int datasetIndex) const{
  if (m_hasSingleOffsets){
    if (m_isMultiBlock){
      for (int i = m_lookUpTable.extent(0); i--;){
//        std::cout << i << "\t" << m_lookUpTable(i,1) << "\t" << m_lookUpTable(i,2) << "\t -- \t" << boundingBox.top() << "\t" << boundingBox.left() << std::endl;
        const auto& lbp = m_extractors[m_lookUpTable(i,0)];
        try {
          dataset(datasetIndex,i) = lbp->extract(m_integralImage, boundingBox.itop() + m_lookUpTable(i,1), boundingBox.ileft() + m_lookUpTable(i,2), true);
        } catch (std::runtime_error& e){
          std::cerr << "Couldn't extract feature from bounding box " << boundingBox.itop() << "," << boundingBox.ileft() << "," << boundingBox.ibottom() << "," <<boundingBox.iright() << " with extractor " << lbp->getBlockSize()[0] << "," << lbp->getBlockSize()[1] << " at position [" << m_lookUpTable(i,1) << "," << m_lookUpTable(i,2) << "]" << std::endl;
          throw;
        }
      }
    } else {
      for (int i = m_lookUpTable.extent(0); i--;){
        const auto& lbp = m_extractors[m_lookUpTable(i,0)];
        dataset(datasetIndex,i) = lbp->extract(m_image, boundingBox.itop() + m_lookUpTable(i,1), boundingBox.ileft() + m_lookUpTable(i,2));
      }
    }
  } else {
    // extract full feature set
    if (m_isMultiBlock){
      blitz::Array<double,2> subwindow = m_integralImage(blitz::Range(boundingBox.itop(), boundingBox.ibottom()), blitz::Range(boundingBox.ileft(), boundingBox.iright()));
      for (int e = 0; e < (int)m_extractors.size(); ++e){
        m_extractors[e]->extract(subwindow, m_featureImages[e], true);
      }
    } else {
      blitz::Array<double,2> subwindow = m_image(blitz::Range(boundingBox.itop(), boundingBox.ibottom()-1), blitz::Range(boundingBox.ileft(), boundingBox.iright()-1));
      for (int e = 0; e < (int)m_extractors.size(); ++e){
        m_extractors[e]->extract(subwindow, m_featureImages[e], false);
      }
    }
    // copy data back to the dataset
    for (int e = 0; e < (int)m_extractors.size(); ++e){
      blitz::Array<uint16_t,1> data_slice = dataset(datasetIndex, blitz::Range(m_featureStarts(e), m_featureStarts(e+1)-1));
      blitz::Array<uint16_t,1>::iterator dit = data_slice.begin();
      blitz::Array<uint16_t,2>::iterator fit = m_featureImages[e].begin();
      blitz::Array<uint16_t,2>::iterator fit_end = m_featureImages[e].end();
      std::copy(fit, fit_end, dit);
    }
  }
}

void bob::ip::facedetect::FeatureExtractor::extractSome(const BoundingBox& boundingBox, blitz::Array<uint16_t,1>& featureVector) const{
  if (m_modelIndices.extent(0) == 0)
    throw std::runtime_error("Please set the model indices before calling this function!");
  // extract only required data
  return extractIndexed(boundingBox, featureVector, m_modelIndices);
}

void bob::ip::facedetect::FeatureExtractor::extractIndexed(const BoundingBox& boundingBox, blitz::Array<uint16_t,1>& featureVector, const blitz::Array<int32_t,1>& indices) const{
  if (indices.extent(0) == 0)
    throw std::runtime_error("The given indices are empty!");
  // extract only requested data
  if (m_isMultiBlock){
    for (int i = indices.extent(0); i--;){
      int index = indices(i);
      const auto& lbp = m_extractors[m_lookUpTable(index,0)];
      featureVector(index) = lbp->extract(m_integralImage, boundingBox.top() + m_lookUpTable(index,1), boundingBox.left() + m_lookUpTable(index,2), true);
    }
  } else {
    for (int i = indices.extent(0); i--;){
      int index = indices(i);
      const auto& lbp = m_extractors[m_lookUpTable(index,0)];
      featureVector(index) = lbp->extract(m_image, boundingBox.top() + m_lookUpTable(index,1), boundingBox.left() + m_lookUpTable(index,2));
    }
  }
}

void bob::ip::facedetect::FeatureExtractor::load(bob::io::base::HDF5File& hdf5file){
  // get global information
  m_patchSize[0] = hdf5file.read<int32_t>("PatchSize", 0);
  m_patchSize[1] = hdf5file.read<int32_t>("PatchSize", 1);

  // get the LBP extractors
  m_extractors.clear();
  for (int i = 1;; ++i){
    std::string dir = (boost::format("LBP_%d") %i).str();
    if (!hdf5file.hasGroup(dir))
      break;
    hdf5file.cd(dir);
    m_extractors.push_back(boost::shared_ptr<bob::ip::base::LBP>(new bob::ip::base::LBP(hdf5file)));
    hdf5file.cd("..");
  }
  m_isMultiBlock = m_extractors[0]->isMultiBlockLBP();

  m_hasSingleOffsets = hdf5file.contains("SelectedOffsets");
  if (m_hasSingleOffsets){
    m_lookUpTable.reference(hdf5file.readArray<int,2>("SelectedOffsets"));
    m_featureStarts.resize(m_extractors.size()+1);
    m_featureStarts(0) = 0;
    int i = 1, j = 1;
    for (; i < m_lookUpTable.extent(0); ++i){
      if (m_lookUpTable(i-1,0) != m_lookUpTable(i,0))
        m_featureStarts(j++) = i;
    }
    m_featureStarts(j) = m_lookUpTable.extent(0);

    // REMOVE patch images since they are not required!
    m_featureImages.clear();
//    std::cout << "Loaded " << m_lookUpTable.extent(0) << " features extractors" << std::endl;

  } else {
    init();
  }
}

void bob::ip::facedetect::FeatureExtractor::save(bob::io::base::HDF5File& hdf5file) const{
  // set global information
  blitz::Array<int32_t,1> t(2);
  t(0) = m_patchSize[0];
  t(1) = m_patchSize[1];
  hdf5file.setArray("PatchSize", t);

  // set the LBP extractors
  for (unsigned i = 0; i != m_extractors.size(); ++i){
    std::string dir = (boost::format("LBP_%d") % (i+1)).str();
    hdf5file.createGroup(dir);
    hdf5file.cd(dir);
    m_extractors[i]->save(hdf5file);
    hdf5file.cd("..");
  }

  if (m_hasSingleOffsets){
    // write all the offsets as well
    hdf5file.setArray("SelectedOffsets", m_lookUpTable);
  }
}
