#include <Join3KeyPipeline.h>
#include <PDBPageHandle.h>
#include <PDBAbstractPageSet.h>
#include <PDBVector.h>
#include "../../../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlockMeta.h"

void pdb::Join3KeyPipeline::runSide(int32_t node, int32_t set) {

  // get the right page set
  PDBAbstractPageSetPtr pageSet;
  if(set == 0) {
    pageSet = keySourcePageSets0[node];
  }
  else if (set == 1) {
    pageSet = keySourcePageSets1[node];
  }
  else if (set == 2) {
    pageSet = keySourcePageSets2[node];
  }

  std::vector<key> myRecords;
  myRecords.reserve(100);

  // get all the pages from the page set
  PDBPageHandle page;
  while ((page = pageSet->getNextPage(0)) != nullptr) {

    // repin the page
    page->repin();

    // get the record
    auto curRec = (Record<Vector<Handle<matrix::MatrixBlockMeta>>> *) page->getBytes();

    // get the root vector
    auto inVec = curRec->getRootObject();

    // go through the vector
    for(int i = 0; i < inVec->size(); ++i) {
      myRecords.push_back({(int32_t)(*inVec)[i]->rowID, (int32_t)(*inVec)[i]->colID});
    }
  }

  if(set == 0) {

    // lock
    std::unique_lock<std::mutex> lck(m0);

    // insert the records
    for(auto &r : myRecords) {
      nodeRecords0[r] = std::make_pair(tid0++, node);
    }
  }
  else if (set == 1) {

    // lock
    std::unique_lock<std::mutex> lck(m1);

    // insert the records
    for(auto &r : myRecords) {
      nodeRecords1[r] = std::make_pair(tid1++, node);
    }
  }
  else if (set == 2) {

    // lock
    std::unique_lock<std::mutex> lck(m2);

    // insert the records
    for(auto &r : myRecords) {
      nodeRecords2[r] = std::make_pair(tid2++, node);
    }
  }
}

void pdb::Join3KeyPipeline::runJoin() {

  // the joined record
  joined_record r;
  for(auto &a_rec : nodeRecords0) {

    // get the tid and nod for a
    auto [a_tid, a_node] = a_rec.second;

    // find the matching record
    auto [b_tid, b_node] = nodeRecords1.find({ a_rec.first.rowID, a_rec.first.colID })->second;

    // find all the matches with c
    for(int32_t i = 0;; ++i) {

      // a.colID = c.rowID
      auto it = nodeRecords2.find({a_rec.first.colID, i});
      if(it == nodeRecords2.end()) {
        break;
      }

      // get the tid and node
      auto [c_tid, c_node] = it->second;

    }


    //// 7. Insert it
    joined.emplace_back(r);
  }
}

bool pdb::operator==(const Join3KeyPipeline::joined_record &lhs,
                     const Join3KeyPipeline::joined_record &rhs) {
  return lhs.first == rhs.first &&
      lhs.second == rhs.second &&
      lhs.third == rhs.third;
}
bool pdb::operator!=(const Join3KeyPipeline::joined_record &lhs,
                     const Join3KeyPipeline::joined_record &rhs) {
  return !(rhs == lhs);
}

bool pdb::operator==(const pdb::Join3KeyPipeline::key &lhs, const pdb::Join3KeyPipeline::key &rhs) {
  return lhs.rowID == rhs.rowID && lhs.colID == rhs.colID;
}
bool pdb::operator!=(const pdb::Join3KeyPipeline::key &lhs, const pdb::Join3KeyPipeline::key &rhs) {
  return !(rhs == lhs);
}
