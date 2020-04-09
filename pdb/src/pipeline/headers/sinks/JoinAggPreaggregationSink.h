#pragma once

#include "EqualsLambda.h"
#include "ComputeSink.h"
#include "TupleSetMachine.h"
#include "TupleSet.h"
#include "../../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlockMeta.h"
#include <vector>
#include <PipJoinAggPlanResult.h>

namespace pdb {

// runs hashes all of the tuples
template<class KeyType, class ValueType>
class JoinAggPreaggregationSink : public ComputeSink {

 private:

  // the attributes to operate on
  int whichAttToHash;
  int whichAttToAggregate;

  // how many partitions do we have
  size_t numPartitions;

  // the mapping from key to the TID
  PDBPageHandle keyToTIDPage;

  PDBPageHandle planPage;

  //
  Handle<Map<KeyType, uint32_t>> keyToTID;

  // the plan
  Handle<PipJoinAggPlanResult> plan;

  // the number of nodes
  uint64_t numNodes;

  // the number of processing threads
  uint64_t numThreads;

 public:

  JoinAggPreaggregationSink(TupleSpec &inputSchema,
                            TupleSpec &attsToOperateOn,
                            uint64_t numNodes,
                            uint64_t numThreads,
                            const PDBPageHandle &planPage,
                            const PDBPageHandle &keyToTIDPage) : numPartitions(numNodes * numThreads),
                                                                 keyToTIDPage(keyToTIDPage),
                                                                 planPage(planPage),
                                                                 numNodes(numNodes),
                                                                 numThreads(numThreads) {

    // to setup the output tuple set
    TupleSpec empty{};
    TupleSetSetupMachine myMachine(inputSchema, empty);

    // this is the input attribute that we will process
    std::vector<int> matches = myMachine.match(attsToOperateOn);
    whichAttToHash = matches[0];
    whichAttToAggregate = matches[1];

    // repin the key to tid page
    keyToTIDPage->repin();

    // the mapping from key to tid
    auto *record = (Record<Map<KeyType, uint32_t>> *) keyToTIDPage->getBytes();
    keyToTID = record->getRootObject();

    // the plan page
    planPage->repin();

    // the extract the page
    auto *planRecord = (Record<PipJoinAggPlanResult> *) this->planPage->getBytes();
    plan = planRecord->getRootObject();
  }

  ~JoinAggPreaggregationSink() override = default;

  Handle<Object> createNewOutputContainer() override {

    // we simply create a new vector of maps to store the stuff
    Handle<Vector<Handle<Map<KeyType, ValueType>>>> returnVal = makeObject<Vector<Handle<Map<KeyType, ValueType>>>>();

    // create the maps
    for(auto i = 0; i < numPartitions; ++i) {

      // add the map
      returnVal->push_back(makeObject<Map<KeyType, ValueType>>());
    }

    // return the output container
    return returnVal;
  }

  void writeOut(TupleSetPtr input, Handle<Object> &writeToMe) override {

    // cast the thing to the map of maps
    Handle<Vector<Handle<Map<KeyType, ValueType>>>> vectorOfMaps = unsafeCast<Vector<Handle<Map<KeyType, ValueType>>>>(writeToMe);

    // get the input columns
    std::vector<KeyType> &keyColumn = input->getColumn<KeyType>(whichAttToHash);
    std::vector<ValueType> &valueColumn = input->getColumn<ValueType>(whichAttToAggregate);

    // and aggregate everyone
    size_t length = keyColumn.size();
    for (size_t i = 0; i < length; i++) {

      // hash the key
      auto hash = hashHim(keyColumn[i]);

      auto s = (pdb::matrix::MatrixBlockMeta *)&keyColumn[i];
      //std::cout << "aggregated " << s->rowID << ", " << s->colID << '\n';

      // figure out the tid of the key and where it is supposed to go
      int32_t tid = (*keyToTID)[keyColumn[i]];
      int32_t node = (*plan->aggToNode)[tid];

      // get the map we are adding to
      Map<KeyType, ValueType> &myMap = (*(*vectorOfMaps)[node * numThreads + (hash % numThreads)]);

      // if this key is not already there...
      if (myMap.count(keyColumn[i]) == 0) {

        // this point will record where the value is located
        ValueType *temp = nullptr;

        // try to add the key... this will cause an allocation for a new key/val pair
        try {

          // get the location that we need to write to...
          temp = &(myMap[keyColumn[i]]);

          // if we get an exception, then we could not fit a new key/value pair
        } catch (NotEnoughSpace &n) {

          // if we got here, then we ran out of space, and so we need to delete the already-processed
          // data so that we can try again...
          keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);
          valueColumn.erase(valueColumn.begin(), valueColumn.begin() + i);
          throw n;
        }

        // we were able to fit a new key/value pair, so copy over the value
        try {
          *temp = valueColumn[i];

          // if we could not fit the value...
        } catch (NotEnoughSpace &n) {

          // then we need to erase the key from the map
          myMap.setUnused(keyColumn[i]);

          // and erase all of these guys from the tuple set since they were processed
          keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);
          valueColumn.erase(valueColumn.begin(), valueColumn.begin() + i);
          throw n;
        }

        // the key is there
      } else {

        // get the value and a copy of it
        ValueType &temp = myMap[keyColumn[i]];
        ValueType copy = temp;

        // and add to the old value, producing a new one
        try {
          temp = copy + valueColumn[i];

          // if we got here, then it means that we ram out of RAM when we were trying
          // to put the new value into the hash table
        } catch (NotEnoughSpace &n) {

          // restore the old value
          temp = copy;

          // and erase all of the guys who were processed
          keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);
          valueColumn.erase(valueColumn.begin(), valueColumn.begin() + i);
          throw n;
        }
      }
    }
  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override { throw runtime_error("JoinAggPreaggregationSink can not write out a page."); }

  // returns the number of records in the preaggregation sink
  uint64_t getNumRecords(Handle<Object> &writeToMe) override {

    // cast the thing to the map of maps
    Handle<Vector<Handle<Map<KeyType, ValueType>>>> vectorOfMaps = unsafeCast<Vector<Handle<Map<KeyType, ValueType>>>>(writeToMe);

    // sum up all the records from each join map
    uint64_t numRecords{0};
    for(int i = 0; i < vectorOfMaps->size(); ++i) {
      numRecords += (*vectorOfMaps)[i]->size();
    }

    // return the size
    return numRecords;
  }
};

}