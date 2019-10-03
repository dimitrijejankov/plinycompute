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
  int whichAttToHash;

  // how many partitions do we have
  size_t numPartitions;

 public:

  JoinAggSink(TupleSpec &inputSchema, TupleSpec &attsToOperateOn, size_t numPartitions) : numPartitions(numPartitions) {

    // to setup the output tuple set
    TupleSpec empty{};
    TupleSetSetupMachine myMachine(inputSchema, empty);

    // this is the input attribute that we will process
    std::vector<int> matches = myMachine.match(attsToOperateOn);
    whichAttToHash = matches[0];
  }

  ~JoinAggSink() override = default;

  Handle<Object> createNewOutputContainer() override {

    // we simply create a new vector of maps to store the stuff
    Handle<Vector<KeyType>> returnVal = makeObject<Vector<KeyType>>();

    // return the output container
    return returnVal;
  }

  void writeOut(TupleSetPtr input, Handle<Object> &writeToMe) override {

    // cast the thing to the map of maps
    Handle<Vector<KeyType>> vectorOfMaps = unsafeCast<Vector<KeyType>>(writeToMe);

    // get the input columns
    std::vector<KeyType> &keyColumn = input->getColumn<KeyType>(whichAttToHash);

    // and aggregate everyone
    size_t length = keyColumn.size();
    for (size_t i = 0; i < length; i++) {

      // hash the key
      auto hash = hashHim(keyColumn[i]);
    }
  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override { throw runtime_error("JoinAggSink can not write out a page."); }

};

}
