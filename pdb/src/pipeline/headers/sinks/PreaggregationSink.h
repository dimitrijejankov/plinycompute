//
// Created by dimitrije on 3/26/19.
//
#include "EqualsLambda.h"
#include "ComputeSink.h"
#include "TupleSetMachine.h"
#include "TupleSet.h"
#include <vector>

#ifndef PDB_PREAGGREGATIONSINK_H
#define PDB_PREAGGREGATIONSINK_H

namespace pdb {

// runs hashes all of the tuples
template<class KeyType, class ValueType>
class PreaggregationSink : public ComputeSink {

 private:

  // the attributes to operate on
  int whichAttToHash;
  int whichAttToAggregate;

  // how many partitions do we have
  size_t numPartitions;

 public:

  PreaggregationSink(TupleSpec &inputSchema, TupleSpec &attsToOperateOn, size_t numPartitions) : numPartitions(numPartitions) {

    // to setup the output tuple set
    TupleSpec empty{};
    TupleSetSetupMachine myMachine(inputSchema, empty);

    // this is the input attribute that we will process
    std::vector<int> matches = myMachine.match(attsToOperateOn);
    whichAttToHash = matches[0];
    whichAttToAggregate = matches[1];
  }

  ~PreaggregationSink() override = default;

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

      // get the map we are adding to
      Map<KeyType, ValueType> &myMap = (*(*vectorOfMaps)[hash % numPartitions]);

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

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override { throw runtime_error("PreaggregationSink can not write out a page."); }

};

}

#endif //PDB_PREAGGREGATIONSINK_H
