#pragma once

#include "executors/ComputeExecutor.h"
#include "TupleSetMachine.h"
#include "TupleSet.h"

namespace pdb {

class UnionExecutor : public ComputeExecutor {

 private:

  // this is the output TupleSet that we return
  TupleSetPtr output;

  // the attribute to operate on
  int whichAtt;

  // the output attribute
  int outAtt;

  // to setup the output tuple set
  TupleSetSetupMachine myMachine;

 public:

  UnionExecutor(TupleSpec &inputSchema, TupleSpec &attsToOperateOn) : myMachine(inputSchema) {

    // this is the input attribute that we will process
    output = std::make_shared<TupleSet>();
    std::vector<int> matches = myMachine.match(attsToOperateOn);
    whichAtt = matches[0];
    outAtt = 0;
  }

  TupleSetPtr process(TupleSetPtr input) override {

    // set up the output tuple set
    myMachine.setup(input, output);

    // copy the column, will overwrite the old one if necessary
    output->copyColumn(input, whichAtt, outAtt);

    // return the output
    return output;
  }
};

}