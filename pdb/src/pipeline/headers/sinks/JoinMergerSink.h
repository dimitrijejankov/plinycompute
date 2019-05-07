//
// Created by dimitrije on 2/19/19.
//

#ifndef PDB_JoinMergerSink_H
#define PDB_JoinMergerSink_H

#include <ComputeSink.h>
#include <TupleSpec.h>
#include <TupleSetMachine.h>
#include <JoinMap.h>
#include <JoinTuple.h>

namespace pdb {

// this class is used to create a ComputeSink object that stores special objects that wrap up multiple columns of a tuple
template<typename RHSType>
class JoinMergerSink : public ComputeSink {

private:

  // the number of partitions
  size_t numPartitions;

  // the worker id
  uint64_t workerID;

public:

  explicit JoinMergerSink(uint64_t workerID, size_t numPartitions) : numPartitions(numPartitions), workerID(workerID) {}

  ~JoinMergerSink() override = default;

  Handle<Object> createNewOutputContainer() override {

    // we simply create a map to hold everything
    Handle<JoinMap<RHSType>> returnVal = makeObject<JoinMap<RHSType>>();
    returnVal->setHashValue(workerID);
    std::cout<<"When calling createNewOutputContainer() in JoinMergerSink, Hash Value for JoinMap is: "<< returnVal->getHashValue() << std::endl;
    return returnVal;
  }

  void writeOut(TupleSetPtr input, Handle<Object> &writeToMe) override {
    throw runtime_error("Join sink can not write out a page.");
  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override {
    // cast the hash table we are merging to
    Handle<JoinMap<RHSType>> mergeToMe = unsafeCast<JoinMap<RHSType>>(writeToMe);

    JoinMap<RHSType> &myMap = *mergeToMe;
    // grab the hash table
    Handle<Object> hashTable = ((Record<Object> *) page->getBytes())->getRootObject();

    auto &joinMapVector = (*unsafeCast<Vector<Handle<JoinMap<RHSType>>>>(hashTable));

    auto &mergeMe = *(joinMapVector[workerID]);

    for (auto it = mergeMe.begin(); it != mergeMe.end(); ++it) {
      // get the records
      JoinRecordList<RHSType> &records = *(*it);
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

};

}

#endif //PDB_JoinMergerSink_H
