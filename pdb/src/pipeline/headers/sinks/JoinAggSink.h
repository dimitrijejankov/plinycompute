#pragma once

#include "EqualsLambda.h"
#include "ComputeSink.h"
#include "TupleSetMachine.h"
#include "TupleSet.h"
#include <vector>


namespace pdb {

// runs hashes all of the tuples
template<class KeyType>
class JoinAggSink : public ComputeSink {
private:

  // the attributes to operate on
  int aggKeyAttribute;
  int leftKeyTIDAttribute;
  int rightKeyTIDAttribute;

  // the first integer is the join key value identifier, the second value is the node it comes from
  using TIDType = std::pair<uint32_t, int32_t>;

  // this is a vector of keys that are joined to form an aggregation group
  using TIDVector = Vector<std::pair<TIDType, TIDType>>;

  // each aggregation group is identified by an unsigned integer
  // this maps maps the aggregation grup to all the keys that are joined to form it
  using TIDIndexMap = Map<uint32_t, TIDVector>;

  // the tids for the aggregation key
  Map<KeyType, uint32_t> aggTIDs;

  // make sure we only create one one join aggregation map
  bool mapCreated = false;

 public:

  JoinAggSink(TupleSpec &outputSchema, TupleSpec &attsToOperateOn, TupleSpec &projection) {

    // to setup the output tuple set
    TupleSpec empty{};
    TupleSetSetupMachine myMachine(outputSchema, empty);

    // get the columns generated
    TupleSpec generated = TupleSpec::complement(outputSchema, projection);

    // this is the input attribute that we will process
    std::vector<int> matches = myMachine.match(generated);
    aggKeyAttribute = matches[0];

    // set the keys
    matches = myMachine.match(projection);
    leftKeyTIDAttribute = matches[0];
    rightKeyTIDAttribute = matches[1];
  }

  ~JoinAggSink() override = default;

  Handle<Object> createNewOutputContainer() override {

    // make sure we only create the map once
    if(mapCreated) {
      throw runtime_error("Already created output map");
    }

    // mark that the map is created
    mapCreated = true;

    // we simply create a new vector of maps to store the stuff
    Handle<TIDIndexMap> returnVal = makeObject<TIDIndexMap>();

    // return the output container
    return returnVal;
  }

  void writeOut(TupleSetPtr input, Handle<Object> &writeToMe) override {

    // the current TID
    uint32_t currentTID = 0;

    // cast the thing to the map of maps
    Handle<TIDIndexMap> outputMap = unsafeCast<TIDIndexMap>(writeToMe);

    // get the input columns
    std::vector<KeyType> &keyColumn = input->getColumn<KeyType>(aggKeyAttribute);
    std::vector<TIDType> &leftTIDColumn = input->getColumn<TIDType>(leftKeyTIDAttribute);
    std::vector<TIDType> &rightTIDColumn = input->getColumn<TIDType>(rightKeyTIDAttribute);

    // and aggregate everyone
    size_t length = keyColumn.size();
    for (size_t i = 0; i < length; i++) {

      // the tid of this aggregation
      uint32_t myTID;

      // try to find the tid for the key
      if(aggTIDs.count(keyColumn[i]) == 0) {

        // get my TID
        myTID = currentTID++;

        // assign a TID to it
        aggTIDs[keyColumn[i]] = myTID;
      }
      else {
        myTID = aggTIDs[keyColumn[i]];
      }

      // store the pair
      (*outputMap)[myTID].push_back({ leftTIDColumn[i], rightTIDColumn[i] });
    }

  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override { throw runtime_error("JoinAggSink can not write out a page."); }

  // returns the number of records in the join agg sink
  uint64_t getNumRecords(Handle<Object> &writeToMe) override {

    // cast the thing to the map of maps
    Handle<TIDIndexMap> outputMap = unsafeCast<TIDIndexMap>(writeToMe);

    // return the size
    return outputMap->size();
  }
};

}
