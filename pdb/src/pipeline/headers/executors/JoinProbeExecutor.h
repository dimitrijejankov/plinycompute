//
// Created by dimitrije on 2/19/19.
//

#ifndef PDB_JOINPROBEEXECUTOR_H
#define PDB_JOINPROBEEXECUTOR_H

#include <TupleSetMachine.h>
#include <TupleSpec.h>
#include <JoinTuple.h>
#include <thread>
#include "ComputeExecutor.h"
#include "StringIntPair.h"
#include "JoinMap.h"
#include "PDBAbstractPageSet.h"

namespace pdb {

// this class is used to encapsulte the computation that is responsible for probing a hash table
template<typename RHSType>
class JoinProbeExecution : public ComputeExecutor {

 private:

  // this is the output TupleSet that we return
  TupleSetPtr output;

  // the attribute to operate on
  int whichAtt;

  // to setup the output tuple set
  TupleSetSetupMachine myMachine;

  // the hash table we are processing
  std::vector<Handle<JoinMap<RHSType>>> inputTables;

  // the pages that contain the hash tables
  std::vector<PDBPageHandle> pages;

  // the list of counts for matches of each of the input tuples
  std::vector<uint32_t> counts;

  // how many nodes are there
  uint64_t numNodes;

  // how many processing threads are there
  uint64_t numProcessingThreads;

  // how many partitions are there
  uint64_t numPartitions;

  // the worker id
  uint64_t workerID;

  // this is the list of all of the output columns in the output TupleSetPtr
  void **columns;

  // used to create space of attributes in the case that the atts from attsToIncludeInOutput are not the first bunch of atts
  // inside of the output tuple
  int offset;

 public:

  ~JoinProbeExecution() {
    if (columns != nullptr)
      delete [] columns;
  }

  // when we probe a hash table, a subset of the atts that we need to put into the output stream are stored in the hash table... the positions
  // of these packed atts are stored in typesStoredInHash, so that they can be extracted.  inputSchema, attsToOperateOn, and attsToIncludeInOutput
  // are standard for executors: they tell us the details of the input that are streaming in, as well as the identity of the has att, and
  // the atts that will be streamed to the output, from the input.  needToSwapLHSAndRhs is true if it's the case that the atts stored in the
  // hash table need to come AFTER the atts being streamed through the join

  JoinProbeExecution(PDBAbstractPageSetPtr &hashTable,
                     std::vector<int> &positions,
                     TupleSpec &inputSchema,
                     TupleSpec &attsToOperateOn,
                     TupleSpec &attsToIncludeInOutput,
                     uint64_t numNodes,
                     uint64_t numProcessingThreads,
                     uint64_t workerID,
                     bool needToSwapLHSAndRhs) : myMachine(inputSchema, attsToIncludeInOutput),
                                                 numNodes(numNodes),
                                                 numProcessingThreads(numProcessingThreads),
                                                 workerID(workerID),
                                                 numPartitions(numNodes * numProcessingThreads) {

    // grab each page and store the hash table
    PDBPageHandle page;
    inputTables.resize(numProcessingThreads);
    while ((page = hashTable->getNextPage(workerID)) != nullptr) {
      // store the page
      pages.emplace_back(page);
      // repin the page
      page->repin();
      // extract the hash table we've been given
      auto *input = (Record<JoinMap<RHSType>> *) page->getBytes();
      auto inputTable = input->getRootObject();
      inputTables[inputTable->getHashValue()] = inputTable;
    }

    // set up the output tuple
    output = std::make_shared<TupleSet>();
    columns = new void *[positions.size()];
    if (needToSwapLHSAndRhs) {
      offset = (int) positions.size();
      createCols<RHSType>(columns, *output, 0, 0, positions);
      std::cout << "We do need to add the pipelined data to the back end of the output tuples.\n";
    } else {
      offset = 0;
      createCols<RHSType>(columns, *output, attsToIncludeInOutput.getAtts().size(), 0, positions);
    }

    // this is the input attribute that we will hash in order to try to find matches
    std::vector<int> matches = myMachine.match(attsToOperateOn);
    whichAtt = matches[0];
  }

  TupleSetPtr process(TupleSetPtr input) override {

    std::vector<size_t> inputHash = input->getColumn<size_t>(whichAtt);

    // redo the vector of hash counts if it's not the correct size
    if (counts.size() != inputHash.size()) {
      counts.resize(inputHash.size());
    }
    // now, run through and attempt to hash
    int overallCounter = 0;
    for (int i = 0; i < inputHash.size(); i++) {

      // grab the approprate hash table

      JoinMap<RHSType> &inputTableRef = *inputTables[(inputHash[i] % numPartitions) % numProcessingThreads];

      // deal with all of the matches
      auto a = inputTableRef.lookup(inputHash[i]);
      int numHits = (int) a.size();

      for (int which = 0; which < numHits; which++) {
        unpack(a[which], overallCounter, 0, columns);
        overallCounter++;
      }
      // remember how many matches we had
      counts[i] = numHits;
    }

    // truncate if we have extra
    eraseEnd<RHSType>(overallCounter, 0, columns);

    // and finally, we need to relpicate the input data
    myMachine.replicate(input, output, counts, offset);

    // outta here!
    return output;
  }
};

}

#endif //PDB_JOINPROBEEXECUTOR_H
