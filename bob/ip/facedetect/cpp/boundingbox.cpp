#include "features.h"
#include <bob.core/logging.h>

boost::shared_ptr<bob::ip::facedetect::BoundingBox> bob::ip::facedetect::BoundingBox::overlap(const BoundingBox& other) const{
  // compute intersection rectangle
  double t = std::max(top(), other.top()),
         b = std::min(bottom(), other.bottom()),
         l = std::max(left(), other.left()),
         r = std::min(right(), other.right());
  return boost::shared_ptr<BoundingBox>(new BoundingBox(t, l, b-t, r-l));
}

double bob::ip::facedetect::BoundingBox::similarity(const BoundingBox& other) const{
  // compute intersection rectangle
  double t = std::max(top(), other.top()),
         b = std::min(bottom(), other.bottom()),
         l = std::max(left(), other.left()),
         r = std::min(right(), other.right());

  // no overlap?
  if (l >= r || t >= b) return 0.;

  // compute overlap
  double intersection = (b-t) * (r-l);
  return intersection / (area() + other.area() - intersection);
}

typedef std::pair<double, int> indexer;
// sort descending
bool gt(const indexer& a, const indexer& b){
  return a.first > b.first;
}

void bob::ip::facedetect::pruneDetections(const std::vector<boost::shared_ptr<BoundingBox>>& boxes, const blitz::Array<double, 1>& weights, double threshold, std::vector<boost::shared_ptr<BoundingBox>>& pruned_boxes, blitz::Array<double, 1>& pruned_weights, const int number_of_detections){
  // sort boxes
  std::vector<indexer> sorted(boxes.size());
  for (int i = boxes.size(); i--;){
    sorted[i] = std::make_pair(weights(i), i);
  }
  std::sort(sorted.begin(), sorted.end(), gt);

  std::list<indexer> pruned;
  std::vector<indexer>::const_iterator sit;
  std::list<indexer>::const_iterator pit;

  if (threshold >= 1.){
    // for overlap == 1 (or larger), all detections will be returned, but sorted
    pruned.insert(pruned.end(), sorted.begin(), sorted.end());
  } else {
    // prune detections (attention, this is O(n^2)!)
    for (sit = sorted.begin(); sit != sorted.end(); ++sit){
      for (pit = pruned.begin(); pit != pruned.end(); ++pit){
        if (boxes[pit->second]->similarity(*boxes[sit->second]) > threshold) break;
      }
      if (pit == pruned.end()){
        pruned.push_back(*sit);
        if (number_of_detections > 0 && pruned.size() == (unsigned)number_of_detections){
          break;
        }
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

void bob::ip::facedetect::groupDetections(const std::vector<boost::shared_ptr<BoundingBox>>& boxes, const blitz::Array<double, 1>& weights, double overlap_threshold, double weight_threshold, unsigned box_count_threshold, std::vector<std::vector<boost::shared_ptr<BoundingBox>>>& grouped_boxes, std::vector<blitz::Array<double, 1>>& grouped_weights){
  if (boxes.empty()){
    bob::core::error << "Cannot find any box to compute overlaps" << std::endl;
    return;
  }
  // sort boxes
  std::vector<indexer> sorted(boxes.size());
  for (int i = boxes.size(); i--;){
    sorted[i] = std::make_pair(weights(i), i);
  }
  std::sort(sorted.begin(), sorted.end(), gt);

  // compute all overlapping detections
  // **this is O(n^2)!**
  std::list<std::list<indexer> > collected;
  std::list<indexer> best;
  best.push_back(sorted.front());
  collected.push_back(best);

  std::vector<indexer>::const_iterator sit = sorted.begin();
  std::list<std::list<indexer> >::iterator cit;
  for (++sit; sit != sorted.end(); ++sit){
    std::list<std::list<indexer> >::iterator best_cit = collected.end();
    double best_overlap = overlap_threshold, current_overlap;

    if (sit->first < weight_threshold)
      // we have reached our weight limit; do not consider more bounding boxes
      break;

    // check if there is a good-enough overlap with one of the already collected bounding boxes
    for (cit = collected.begin(); cit != collected.end(); ++cit){
      current_overlap = boxes[sit->second]->similarity(*boxes[cit->front().second]);
      if (current_overlap > best_overlap){
        // get the bounding box with the highest overlap value
        best_overlap = current_overlap;
        best_cit = cit;
      }
    }

    if (best_cit == collected.end()){
      // no such overlap was found, add a new list of bounding boxes
      std::list<indexer> novel;
      novel.push_back(*sit);
      collected.push_back(novel);
    } else {
      // add the bounding box to the list with the highest overlap
      best_cit->push_back(*sit);
    }
  }


  // now, convert lists to resulting grouped vectors of vectors of bounding boxes
  grouped_boxes.reserve(collected.size());
  grouped_weights.reserve(collected.size());

  std::list<indexer>::const_iterator oit;
  for (cit = collected.begin(); cit != collected.end(); ++cit){
    if (cit->size() >= box_count_threshold){
      blitz::Array<double,1> current_weights(cit->size());
      std::vector<boost::shared_ptr<BoundingBox>> current_boxes(cit->size());
      int o = 0;
      for (oit = cit->begin(); oit != cit->end(); ++oit, ++o){
        current_weights(o) = oit->first;
        current_boxes[o] = boxes[oit->second];
      }
      grouped_boxes.push_back(current_boxes);
      grouped_weights.push_back(current_weights);
    }
  }
  // done.
}


void bob::ip::facedetect::bestOverlap(const std::vector<boost::shared_ptr<BoundingBox>>& boxes, const blitz::Array<double, 1>& weights, double overlap_threshold, std::vector<boost::shared_ptr<BoundingBox>>& overlapping_boxes, blitz::Array<double, 1>& overlapping_weights){
  if (boxes.empty()){
    bob::core::error << "Cannot find any box to compute overlaps" << std::endl;
    return;
  }
  // sort boxes
  std::vector<indexer> sorted(boxes.size());
  for (int i = boxes.size(); i--;){
    sorted[i] = std::make_pair(weights(i), i);
  }
  std::sort(sorted.begin(), sorted.end(), gt);

  std::list<indexer> overlapping;
  std::list<indexer>::const_iterator oit;

  // compute all overlapping detections
  // **this is O(n^2)!**
  std::list<std::list<indexer> > collected;
  std::list<indexer> best;
  best.push_back(sorted.front());
  collected.push_back(best);

  std::vector<indexer>::const_iterator sit = sorted.begin();
  std::list<std::list<indexer> >::iterator cit;
  for (++sit; sit != sorted.end(); ++sit){
    for (cit = collected.begin(); cit != collected.end(); ++cit){
      if (boxes[sit->second]->similarity(*boxes[cit->front().second]) > overlap_threshold){
        cit->push_back(*sit);
        break;
      }
    }
    if (cit == collected.end()){
      std::list<indexer> novel;
      novel.push_back(*sit);
      collected.push_back(novel);
    }
  }

  // now, take the list with the highest TOTAL detection value
  double best_total = 0.;
  for (cit = collected.begin(); cit != collected.end(); ++cit){
    double current_total = 0.;
    for (oit = cit->begin(); oit != cit->end(); ++oit){
      current_total += std::max(oit->first, 0.);
    }
    if (current_total > best_total){
      best_total = current_total;
      overlapping = *cit;
    }
  }

  // fill overlapping boxes
  overlapping_boxes.reserve(overlapping.size());
  overlapping_weights.resize(overlapping.size());
  int i = 0;
  for (oit = overlapping.begin(); oit != overlapping.end(); ++oit, ++i){
    overlapping_boxes.push_back(boxes[oit->second]);
    overlapping_weights(i) = oit->first;
  }

  // done.
}
