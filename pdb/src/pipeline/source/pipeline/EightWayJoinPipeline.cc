#include <EightWayJoinPipeline.h>
#include <PDBPageHandle.h>
#include <PDBAbstractPageSet.h>
#include <PDBVector.h>
#include "../../../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlockMeta3D.h"

void pdb::EightWayJoinPipeline::runSide(int node) {

  //
  auto pageSet = keySourcePageSets[node];

  std::vector<key> myRecords;
  myRecords.reserve(100);

  // get all the pages from the page set
  PDBPageHandle page;
  while ((page = pageSet->getNextPage(0)) != nullptr) {

    // repin the page
    page->repin();

    // get the record
    auto curRec = (Record<Vector<Handle<matrix_3d::MatrixBlockMeta3D>>> *) page->getBytes();

    // get the root vector
    auto inVec = curRec->getRootObject();

    // go through the vector
    for(int i = 0; i < inVec->size(); ++i) {
      myRecords.push_back({(int32_t)(*inVec)[i]->x_id, (int32_t)(*inVec)[i]->y_id, (int32_t)(*inVec)[i]->z_id});
    }
  }

  // lock
  std::unique_lock<std::mutex> lck(m);

  // insert the records
  for(auto &r : myRecords) {
    nodeRecords[r] = std::make_pair(tid++, node);
  }
}

void pdb::EightWayJoinPipeline::runJoin() {

  // the joined record
  joined_record r;
  for(auto top_left_front = nodeRecords.begin(); top_left_front != nodeRecords.end(); ++top_left_front) {

    /// 0. Do it for the top left right

    // set the tid for this one
    r.first = top_left_front->second.first;

    /// 1. Do it for the top right front

    // get the top right front
    auto top_right_front = top_left_front->first;
    top_right_front.first++;

    // check if we have the top right front
    auto it = nodeRecords.find(top_right_front);
    if(it == nodeRecords.end()) {
      continue;
    }
    r.second = it->second.first;


    /// 2. Do it for the bottom left front

    // get the top right front
    auto bottom_left_front = top_left_front->first;
    bottom_left_front.second--;

    // check if we have the top right front
    it = nodeRecords.find(bottom_left_front);
    if(it == nodeRecords.end()) {
      continue;
    }
    r.third = it->second.first;

    /// 3. Do it for the bottom right front

    auto bottom_right_front = top_right_front;
    bottom_right_front.second--;

    // check if we have the top right front
    it = nodeRecords.find(bottom_right_front);
    if(it == nodeRecords.end()) {
      continue;
    }
    r.fourth = it->second.first;

    /// 4. Do it for the top left back

    auto top_left_back = top_left_front->first;
    top_left_back.third--;

    // check if we have the top right front
    it = nodeRecords.find(top_left_back);
    if(it == nodeRecords.end()) {
      continue;
    }
    r.fifth = it->second.first;

    /// 4. Do it for the top right back

    auto top_right_back = top_right_front;
    top_right_back.third--;

    // check if we have the top right front
    it = nodeRecords.find(top_right_back);
    if(it == nodeRecords.end()) {
      continue;
    }
    r.sixth = it->second.first;

    /// 5. Do it for the bottom left back

    auto bottom_left_back = bottom_left_front;
    bottom_left_back.third--;

    // check if we have the top right front
    it = nodeRecords.find(bottom_left_back);
    if(it == nodeRecords.end()) {
      continue;
    }
    r.seventh = it->second.first;


    /// 6. Do it for the bottom right front

    auto bottom_right_back = bottom_right_front;
    bottom_right_back.third--;

    // check if we have the top right front
    it = nodeRecords.find(bottom_right_back);
    if(it == nodeRecords.end()) {
      continue;
    }
    r.eight = it->second.first;

    //// 7. Insert it
    joined.emplace_back(r);
  }
}

bool pdb::operator==(const pdb::EightWayJoinPipeline::joined_record &lhs,
                     const pdb::EightWayJoinPipeline::joined_record &rhs) {
  return lhs.first == rhs.first &&
      lhs.second == rhs.second &&
      lhs.third == rhs.third &&
      lhs.fourth == rhs.fourth &&
      lhs.fifth == rhs.fifth &&
      lhs.sixth == rhs.sixth &&
      lhs.seventh == rhs.seventh &&
      lhs.eight == rhs.eight;
}
bool pdb::operator!=(const pdb::EightWayJoinPipeline::joined_record &lhs,
                     const pdb::EightWayJoinPipeline::joined_record &rhs) {
  return !(rhs == lhs);
}
bool pdb::operator==(const pdb::EightWayJoinPipeline::key &lhs, const pdb::EightWayJoinPipeline::key &rhs) {
  return lhs.first == rhs.first &&
      lhs.second == rhs.second &&
      lhs.third == rhs.third;
}
bool pdb::operator!=(const pdb::EightWayJoinPipeline::key &lhs, const pdb::EightWayJoinPipeline::key &rhs) {
  return !(rhs == lhs);
}
