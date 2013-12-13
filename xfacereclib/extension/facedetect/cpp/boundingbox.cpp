#include "features.h"

BoundingBox BoundingBox::overlap(const BoundingBox& other) const{
  // compute intersection rectangle
  double t = std::max(top(), other.top()),
         b = std::min(bottom(), other.bottom()),
         l = std::max(left(), other.left()),
         r = std::min(right(), other.right());
  return BoundingBox(t, l, b-t+1, r-l+1);
}

double BoundingBox::similarity(const BoundingBox& other) const{
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

typedef std::pair<double, int> indexer;
// sort descending
bool gt(const indexer& a, const indexer& b){
  return a.first > b.first;
}

void pruneDetections(const std::vector<BoundingBox>& boxes, const blitz::Array<double, 1>& weights, double threshold, std::vector<BoundingBox>& pruned_boxes, blitz::Array<double, 1>& pruned_weights, const int number_of_detections){
  // sort boxes
  std::vector<indexer> sorted(boxes.size());
  for (int i = boxes.size(); i--;){
    sorted[i] = std::make_pair(weights(i), i);
  }
  std::sort(sorted.begin(), sorted.end(), gt);

  // prune detections (attention, this is O(n^2)!)
  std::list<indexer> pruned;
  std::vector<indexer>::const_iterator sit;
  std::list<indexer>::const_iterator pit;
  for (sit = sorted.begin(); sit != sorted.end(); ++sit){
    for (pit = pruned.begin(); pit != pruned.end(); ++pit){
      if (boxes[pit->second].similarity(boxes[sit->second]) > threshold) break;
    }
    if (pit == pruned.end()){
      pruned.push_back(*sit);
      if (number_of_detections > 0 && pruned.size() == (unsigned)number_of_detections){
        break;
      }
    }
  }

  // fill pruned boxes
  pruned_boxes.reserve(pruned.size());
  pruned_weights.resize(pruned.size());
  int i = 0;
  for (pit = pruned.begin(); pit != pruned.end(); ++pit, ++i){
    pruned_boxes.push_back(boxes[pit->second]);
    pruned_weights(i) = pit->first;
  }

  // done.
}

void bestOverlap(const std::vector<BoundingBox>& boxes, const blitz::Array<double, 1>& weights, double threshold, std::vector<BoundingBox>& overlapping_boxes, blitz::Array<double, 1>& overlapping_weights){
  // sort boxes
  std::vector<indexer> sorted(boxes.size());
  for (int i = boxes.size(); i--;){
    sorted[i] = std::make_pair(weights(i), i);
  }
  std::sort(sorted.begin(), sorted.end(), gt);

  // prune detections (attention, this is O(n^2)!)
  std::list<indexer> overlapping;
  BoundingBox best = boxes[sorted.front().second];
  std::vector<indexer>::const_iterator sit;
  std::list<indexer>::const_iterator pit;
  for (sit = sorted.begin(); sit != sorted.end(); ++sit){
    if (boxes[sit->second].similarity(best) > threshold){
      overlapping.push_back(*sit);
    }
  }

  // fill pruned boxes
  overlapping_boxes.reserve(overlapping.size());
  overlapping_weights.resize(overlapping.size());
  int i = 0;
  for (pit = overlapping.begin(); pit != overlapping.end(); ++pit, ++i){
    overlapping_boxes.push_back(boxes[pit->second]);
    overlapping_weights(i) = pit->first;
  }

  // done.
}
