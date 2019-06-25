//
// Created by yuxin on 2/19/19.
//

#ifndef PDB_BROADCASTJOINCOMBINERSINK_H
#define PDB_BROADCASTJOINCOMBINERSINK_H

#include <ComputeSink.h>
#include <TupleSpec.h>
#include <TupleSetMachine.h>
#include <JoinMap.h>
#include <JoinTuple.h>

namespace pdb {

// this class is used to create a ComputeSink object that stores special objects that wrap up multiple columns of a tuple
template<typename RHSType>
class BroadcastJoinCombinerSink : public ComputeSink {

 private:

  //number of nodes
  uint64_t numNodes;

  //number of threads on each node
  uint64_t numThreads;

  // the worker id
  uint64_t workerID;

 public:

  explicit BroadcastJoinCombinerSink(uint64_t workerID, uint64_t numThreads, uint64_t numNodes) : workerID(workerID),numThreads(numThreads), numNodes(numNodes){}

  ~BroadcastJoinCombinerSink() override = default;

  Handle<Object> createNewOutputContainer() override {

    // we simply create a map to hold everything
    Handle<JoinMap<RHSType>> returnVal = makeObject<JoinMap<RHSType>>();
    returnVal->setHashValue(workerID);
    return returnVal;
  }

  void writeOut(TupleSetPtr input, Handle<Object> &writeToMe) override {
    throw runtime_error("Join sink can not write out a page.");
  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override {
    // cast the hash table we are merging to
    Handle<JoinMap<RHSType>> mergeToMe = unsafeCast<JoinMap<RHSType>>(writeToMe);

    JoinMap<RHSType> &myMap = *mergeToMe;

    page->repin();

    // grab the hash table
    Handle<Object> hashTable = ((Record<Object> *) page->getBytes())->getRootObject();

    auto &joinMapVector = (*unsafeCast<Vector<Handle<JoinMap<RHSType>>>>(hashTable));

    for (int offset = 0; offset < numNodes; offset++) {

      // calculate the JoinMap indexes.
      // Combine all the JoinMaps with the same index. Index = NodeID(range from 0~numNodes-1) * numThreads + workerID
      auto joinMapIndex = offset * numThreads + workerID;

      auto &mergeMe = *(joinMapVector[joinMapIndex]);

      for (auto it = mergeMe.begin(); it != mergeMe.end(); ++it) {
        // get the ptr
        auto recordsPtr = *it;

        // get the records
        JoinRecordList<RHSType> records = *(recordsPtr);

        // get the hash
        auto hash = records.getHash();

        // copy the records
        for (size_t i = 0; i < records.size(); ++i) {
          // copy a single record
          try {
            RHSType &temp = myMap.push(hash);
            temp = records[i];
            // if we get an exception, then we could not fit a new key/value pair
          } catch (NotEnoughSpace &n) {
            // this must not happen. The combined records of the partition // TODO maybe handle this gracefully
            throw n;
          }
        }
      }
    }
  }

};

}

#endif //PDB_JoinMergerSink_H
