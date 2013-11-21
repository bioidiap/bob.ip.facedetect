#include "features.h"
#include <boost/format.hpp>

FeatureExtractor::FeatureExtractor(const blitz::TinyVector<int,2>& patchSize, const bob::ip::LBP& templAte, bool overlap, bool square)
: m_patchSize(patchSize),
  m_lookUpTable(0,3),
  m_extractors(),
  m_isMultiBlock(templAte.isMultiBlockLBP())
{
  // initialize the extractors
  if (!m_isMultiBlock){
    for (int dy = 1; dy < patchSize[0] / 2; ++dy)
      for (int dx = 1; dx < patchSize[1] / 2; ++dx)
        if (!square || dy == dx){
          // copy setup from template LBP
          bob::ip::LBP lbp(templAte);
          // adapt radius / block size
          lbp.setRadii(blitz::TinyVector<double,2>(dy, dx));
          // add LBP
          m_extractors.push_back(lbp);
    }
  } else {
    if (overlap){
      for (int dy = 1; dy <= patchSize[0] - 2; ++dy)
        for (int dx = 1; dx <= patchSize[1] - 2; ++dx)
          if (!square || dy == dx){
            // copy setup from template LBP
            bob::ip::LBP lbp(templAte);
            // adapt radius / block size
            lbp.setBlockSizeAndOverlap(blitz::TinyVector<int,2>(dy, dx), blitz::TinyVector<int,2>(dy-1, dx-1));
            // add LBP
            m_extractors.push_back(lbp);
      }
    } else {
      for (int dy = 1; dy <= patchSize[0] / 3; ++dy)
        for (int dx = 1; dx <= patchSize[1] / 3; ++dx)
          if (!square || dy == dx){
            // copy setup from template LBP
            bob::ip::LBP lbp(templAte);
            // adapt radius / block size
            lbp.setBlockSize(blitz::TinyVector<int,2>(dy, dx));
            // add LBP
            m_extractors.push_back(lbp);
      }
    }
  }

  // compute LookUpTable
  init();
}


FeatureExtractor::FeatureExtractor(const blitz::TinyVector<int,2>& patchSize, const std::vector<bob::ip::LBP>& extractors)
: m_patchSize(patchSize),
  m_lookUpTable(0,3),
  m_extractors(extractors)
{
  m_isMultiBlock = extractors[0].isMultiBlockLBP();
  // check if all other lbp extractors have the same multi-block characteristics
  for (std::vector<bob::ip::LBP>::const_iterator it = extractors.begin(); it != extractors.end(); ++it){
    if (it->isMultiBlockLBP() != m_isMultiBlock){
      throw std::runtime_error("All LBP variants need to be multiblock or all are not -- a mix is not possible!");
    }
  }

  init();
}

FeatureExtractor::FeatureExtractor(const FeatureExtractor& other)
: m_patchSize(other.m_patchSize),
  m_lookUpTable(other.m_lookUpTable),
  m_extractors(other.m_extractors),
  m_featureStarts(other.m_featureStarts),
  m_modelIndices(other.m_modelIndices),
  m_isMultiBlock(other.m_isMultiBlock)
{
  // we copy everything, except for the internally allocated memory
  m_featureImages.clear();
  for (int e = 0; e < (int)m_extractors.size(); ++e){
    blitz::TinyVector<int,2> shape = m_extractors[e].getLBPShape(m_patchSize, false);
    m_featureStarts(e+1) = m_featureStarts(e) + shape[0] * shape[1];
    m_featureImages.push_back(blitz::Array<uint16_t, 2>(shape));
  }
}


FeatureExtractor::FeatureExtractor(bob::io::HDF5File& file){
  // read information from file
  load(file);
}

void FeatureExtractor::init(){
  // initialize the indices for the full feature vector extraction
  m_featureStarts.resize(m_extractors.size()+1);
  m_featureStarts(0) = 0;
  m_featureImages.clear();
  for (int e = 0; e < (int)m_extractors.size(); ++e){
    blitz::TinyVector<int,2> shape = m_extractors[e].getLBPShape(m_patchSize, false);
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
        m_lookUpTable(index,1) = y + m_extractors[e].getOffset()[0];
        // offset in x in this extractor
        m_lookUpTable(index,2) = x + m_extractors[e].getOffset()[1];
    }
  }
}


void FeatureExtractor::extract_all(const BoundingBox& boundingBox, blitz::Array<uint16_t,2>& dataset, int datasetIndex) const{
  // extract full feature set
  if (m_isMultiBlock){
    blitz::Array<double,2> subwindow = m_integralImage(blitz::Range(boundingBox.top(), boundingBox.bottom()+1), blitz::Range(boundingBox.left(), boundingBox.right()+1));
    for (int e = 0; e < (int)m_extractors.size(); ++e){
      m_extractors[e](subwindow, m_featureImages[e], true);
    }
  } else {
    blitz::Array<double,2> subwindow = m_image(blitz::Range(boundingBox.top(), boundingBox.bottom()), blitz::Range(boundingBox.left(), boundingBox.right()));
    for (int e = 0; e < (int)m_extractors.size(); ++e){
      m_extractors[e](subwindow, m_featureImages[e], false);
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

void FeatureExtractor::extract_some(const BoundingBox& boundingBox, blitz::Array<uint16_t,1>& featureVector) const{
  // extract only required data
  if (m_modelIndices.extent(0) == 0)
    throw std::runtime_error("Please set the model indices before calling this function!");

  if (m_isMultiBlock){
    for (int i = m_modelIndices.extent(0); i--;){
      int index = m_modelIndices(i);
      const bob::ip::LBP& lbp = m_extractors[m_lookUpTable(index,0)];
      featureVector(index) = lbp.operator()(m_integralImage, boundingBox.top() + m_lookUpTable(index,1), boundingBox.left() + m_lookUpTable(index,2), true);
    }
  } else {
    for (int i = m_modelIndices.extent(0); i--;){
      int index = m_modelIndices(i);
      const bob::ip::LBP& lbp = m_extractors[m_lookUpTable(index,0)];
      featureVector(index) = lbp.operator()(m_image, boundingBox.top() + m_lookUpTable(index,1), boundingBox.left() + m_lookUpTable(index,2));
    }
  }
}

void FeatureExtractor::load(bob::io::HDF5File& hdf5file){
  // get global information
  m_patchSize[0] = hdf5file.read<int64_t>("PatchSize", 0);
  m_patchSize[1] = hdf5file.read<int64_t>("PatchSize", 1);

  // get the LBP extractors
  m_extractors.clear();
  for (int i = 1;; ++i){
    std::string dir = (boost::format("LBP_%d") %i).str();
    if (!hdf5file.hasGroup(dir))
      break;
    hdf5file.cd(dir);
    m_extractors.push_back(bob::ip::LBP(hdf5file));
    hdf5file.cd("..");
  }
  init();
}

void FeatureExtractor::save(bob::io::HDF5File& hdf5file) const{
  // set global information
  blitz::Array<int64_t,1> t(2);
  t(0) = m_patchSize[0];
  t(1) = m_patchSize[1];
  hdf5file.setArray("PatchSize", t);

  // set the LBP extractors
  for (unsigned i = 0; i != m_extractors.size(); ++i){
    std::string dir = (boost::format("LBP_%d") % (i+1)).str();
    hdf5file.createGroup(dir);
    hdf5file.cd(dir);
    m_extractors[i].save(hdf5file);
    hdf5file.cd("..");
  }
}

