#include <utility>
#include <TRABlock.h>

#pragma once

namespace pdb {

template<class IN1, class IN2>
class TRAJoinSource : public ComputeSource {
 public:

  TRAJoinSource(int32_t nodeID,
                int32_t workerID,
                int32_t numWorkers,
                JoinAggTupleEmitterPtr emitter,
                PDBRandomAccessPageSetPtr lhsPageSet,
                PDBRandomAccessPageSetPtr rhsPageSet) : nodeID(nodeID),
                                                        workerID(workerID),
                                                        numWorkers(numWorkers),
                                                        emitter(std::move(emitter)),
                                                        leftInputPageSet(std::move(lhsPageSet)),
                                                        rightInputPageSet(std::move(rhsPageSet)),
                                                        currentRecord(0),
                                                        lastLHSPage(0),
                                                        lastRHSPage(0) {

    // create the tuple set that we'll return during iteration
    output = std::make_shared<TupleSet>();

    // the lhs column
    lhsColumn = new std::vector<Handle<IN1>>;
    output->addColumn(0, lhsColumn, true);

    // the rhs column
    rhsColumn = new std::vector<Handle<IN2>>;
    output->addColumn(1, rhsColumn, true);

    // reserve some records
    records.reserve(1000);
  }

  TupleSetPtr getNextTupleSet(const PDBTupleSetSizePolicy &policy) override {

    // did we manage to process the input
    if (!policy.inputWasProcessed() && output != nullptr) {

        // rollback the records
        currentRecord -= lhsColumn->size();
    }

    // if we are done here return null
    if (records.size() == end) {

      // get the records
      int lhsPage = lastLHSPage;
      int rhsPage = lastRHSPage;
      emitter->getRecords(records, lastLHSPage, lastRHSPage, workerID);

      // we are done if we could not get any new records
      if(records.empty()) {
        return nullptr;
      }

      // we start from record zero
      currentRecord = 0;

      // go through the pages and get the vectors
      for (int i = lhsPage; i < lastLHSPage; ++i) {

        // pin this page
        (*leftInputPageSet)[i]->repin();

        // get the vector from the left page
        auto record = ((Record<Vector<Handle<IN1>>> *) (*leftInputPageSet)[i]->getBytes());

        // store it
        leftTuples.emplace_back(record->getRootObject());
      }

      // go through the pages and the record
      for (int i = rhsPage; i < lastRHSPage; ++i) {

        // pin this page
        (*rightInputPageSet)[i]->repin();

        // get the vector from the right page
        auto record = ((Record<Vector<Handle<IN2>>> *) (*rightInputPageSet)[i]->getBytes());

        // store it
        rightTuples.emplace_back(record->getRootObject());
      }
    }

    // figure out the offset
    end = std::min<uint32_t>(records.size(), currentRecord + policy.getChunksSize());

    // the number of rows we have
    auto numRows = end - currentRecord;

    // resize the columns
    lhsColumn->resize(numRows);
    rhsColumn->resize(numRows);

    // go through all the records we need to copy
    for(int i = currentRecord; i < end; i++) {

      // set the columns
      (*lhsColumn)[i - currentRecord] = (*leftTuples[records[i].lhs_page])[records[i].lhs_record];
      (*rhsColumn)[i - currentRecord] = (*rightTuples[records[i].rhs_page])[records[i].rhs_record];

      if constexpr (std::is_same<IN1, pdb::TRABlock>::value) {
        std::cout << "LHS : ";
          (*leftTuples[records[i].lhs_page])[records[i].lhs_record]->print_meta();
        std::cout << "RHS : ";
          (*rightTuples[records[i].rhs_page])[records[i].rhs_record]->print_meta();
        std::cout << "-------------------------------\n";
      }

      //std::cout << (*lhsColumn)[i - currentRecord]->getKeyRef().key0 << ", " << (*lhsColumn)[i - currentRecord]->getKeyRef().key1 << '\n';
      //std::cout << (*rhsColumn)[i - currentRecord]->getKeyRef().key0 << ", " << (*rhsColumn)[i - currentRecord]->getKeyRef().key1 << '\n';
    }

    // move the current record
    currentRecord += numRows;

    // std::cout << "Emitted " << numRows << " : " << currentRecord << "/" << records.size() << '\n';
    // return the output
    return output;
  }

 private:

  // the records we want to emmit
  std::vector<JoinAggTupleEmitter::JoinedRecord> records;

  // the two columns
  std::vector<Handle<IN1>> *lhsColumn;
  std::vector<Handle<IN2>> *rhsColumn;

  // and the tuple set we return
  TupleSetPtr output;

  // the current record
  uint64_t currentRecord;

  uint64_t end = 0;

  // these are the vectors from where we grab all the tuples
  std::vector<Handle < Vector < Handle < IN1>>>> leftTuples;
  std::vector<Handle < Vector < Handle < IN2>>>> rightTuples;

  // the id of this node
  int32_t nodeID{};

  // the id of the worker
  int32_t workerID{};

  // the number of workers
  int32_t numWorkers{};

  // the join record emitter
  JoinAggTupleEmitterPtr emitter;

  // the plan
  Handle<PipJoinAggPlanResult> plan;

  // the page that contains the plan
  PDBPageHandle planPage;

  // the left page set
  PDBRandomAccessPageSetPtr leftInputPageSet;

  // the right page set
  PDBRandomAccessPageSetPtr rightInputPageSet;

  //
  int32_t lastLHSPage;

  int32_t lastRHSPage;
};

}