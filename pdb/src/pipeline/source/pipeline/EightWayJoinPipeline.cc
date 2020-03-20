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
      myRecords.emplace_back(std::make_tuple((*inVec)[i]->x_id, (*inVec)[i]->y_id, (*inVec)[i]->z_id));
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
    get<0>(r) = top_left_front->second.first;

    /// 1. Do it for the top right front

    // get the top right front
    auto top_right_front = top_left_front->first;
    get<0>(top_right_front)++;

    // check if we have the top right front
    auto it = nodeRecords.find(top_right_front);
    if(it == nodeRecords.end()) {
      continue;
    }
    get<1>(r) = it->second.first;


    /// 2. Do it for the bottom left front

    // get the top right front
    auto bottom_left_front = top_left_front->first;
    get<1>(bottom_left_front)--;

    // check if we have the top right front
    it = nodeRecords.find(bottom_left_front);
    if(it == nodeRecords.end()) {
      continue;
    }
    get<2>(r) = it->second.first;

    /// 3. Do it for the bottom right front

    auto bottom_right_front = top_right_front;
    get<1>(bottom_right_front)--;

    // check if we have the top right front
    it = nodeRecords.find(bottom_right_front);
    if(it == nodeRecords.end()) {
      continue;
    }
    get<3>(r) = it->second.first;

    /// 4. Do it for the top left back

    auto top_left_back = top_left_front->first;
    get<2>(top_left_back)--;

    // check if we have the top right front
    it = nodeRecords.find(top_left_back);
    if(it == nodeRecords.end()) {
      continue;
    }
    get<4>(r) = it->second.first;

    /// 4. Do it for the top right back

    auto top_right_back = top_right_front;
    get<2>(top_right_back)--;

    // check if we have the top right front
    it = nodeRecords.find(top_right_back);
    if(it == nodeRecords.end()) {
      continue;
    }
    get<5>(r) = it->second.first;

    /// 5. Do it for the bottom left back

    auto bottom_left_back = bottom_left_front;
    get<2>(bottom_left_back)--;

    // check if we have the top right front
    it = nodeRecords.find(bottom_left_back);
    if(it == nodeRecords.end()) {
      continue;
    }
    get<6>(r) = it->second.first;


    /// 6. Do it for the bottom right front

    auto bottom_right_back = bottom_right_front;
    get<2>(bottom_right_back)--;

    // check if we have the top right front
    it = nodeRecords.find(bottom_right_back);
    if(it == nodeRecords.end()) {
      continue;
    }
    get<7>(r) = it->second.first;

    //// 7. Insert it
    joined.emplace_back(r);
  }
}
