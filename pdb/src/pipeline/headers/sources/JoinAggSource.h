#include <utility>

#pragma once

namespace pdb {

template<class IN1, class IN2>
class JoinAggSource : public ComputeSource {
 public:

  JoinAggSource(int32_t nodeID,
                int32_t workerID,
                int32_t numWorkers,
                std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &leftTIDToRecordMapping,
                std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &rightTIDToRecordMapping,
                const PDBPageHandle &page,
                PDBRandomAccessPageSetPtr lhsPageSet,
                PDBRandomAccessPageSetPtr rhsPageSet) : nodeID(nodeID),
                                                               workerID(workerID),
                                                               numWorkers(numWorkers),
                                                               planPage(page),
                                                               leftTIDToRecordMapping(leftTIDToRecordMapping),
                                                               rightTIDToRecordMapping(rightTIDToRecordMapping),
                                                               leftInputPageSet(std::move(lhsPageSet)),
                                                               rightInputPageSet(std::move(rhsPageSet)) {

    // get the plan result so we can do stuff
    page->repin();
    auto *recordCopy = (Record<PipJoinAggPlanResult> *) this->planPage->getBytes();
    plan = recordCopy->getRootObject();

    // extract the input vectors
    leftInputPageSet->repinAll();
    for (int i = 0; i < leftInputPageSet->getNumPages(); ++i) {

      //
      auto record = ((Record<Vector<std::pair<uint32_t, Handle<IN1>>>> *) (*leftInputPageSet)[i]->getBytes());

      // the
      leftTuples.emplace_back(record->getRootObject());
    }


    // extract the input vectors
    rightInputPageSet->repinAll();
    for (int i = 0; i < rightInputPageSet->getNumPages(); ++i) {

      //
      auto record = ((Record<Vector<std::pair<uint32_t, Handle<IN2>>>> *) (*rightInputPageSet)[i]->getBytes());

      // the
      rightTuples.emplace_back(record->getRootObject());
    }

    // the join group nodes
    allJoinGroups = &((*plan->joinGroupsPerNode)[nodeID]);

    // how much per worker
    int stride = (int32_t) allJoinGroups->size() / numWorkers;

    // figure out the start and the end
    joinGroupStart = workerID * stride;
    joinGroupEnd = (workerID + 1) * stride;

    // if this is the last worker
    joinGroupEnd += workerID == (numWorkers - 1) ? (int32_t) allJoinGroups->size() % numWorkers : 0;

    // we start from record zero
    currentRecord = 0;

    // create the tuple set that we'll return during iteration
    output = std::make_shared<TupleSet>();

    // the lhs column
    lhsColumn = new std::vector<Handle<IN1>>;
    output->addColumn(0, lhsColumn, true);

    // the lhs column
    rhsColumn = new std::vector<Handle<IN2>>;
    output->addColumn(1, rhsColumn, true);
  }

  TupleSetPtr getNextTupleSet(const PDBTupleSetSizePolicy &policy) override {

    // did we manage to process the input
    if (!policy.inputWasProcessed() && output != nullptr) {

        // rollback the records
        currentRecord -= lhsColumn->size();
    }

    // figure out the offset
    auto start = joinGroupStart + currentRecord;
    auto end = std::min(joinGroupEnd, start + policy.getChunksSize());

    // if we are done here return null
    if (start >= end) {
      return nullptr;
    }

    // the number of rows we have
    auto numRows = end - start;

    // resize the columns
    lhsColumn->resize(numRows);
    rhsColumn->resize(numRows);

    // go through all the records we need to copy
    for (auto i = start; i < end; ++i) {

      // get the tids
      auto leftTID = (*allJoinGroups)[i].first;
      auto rightTID = (*allJoinGroups)[i].second;

      // find all the left ones we ned to join
      auto [lhsPage, lhsRecord] = findLeftRecord(leftTID);
      auto [rhsPage, rhsRecord] = findRightRecord(rightTID);

      // set the columns
      (*lhsColumn)[i - start] = (*leftTuples[lhsPage])[lhsRecord].second;
      (*rhsColumn)[i - start] = (*rightTuples[rhsPage])[rhsRecord].second;
    }

    // move the current record
    currentRecord += numRows;

    // return the output
    return output;
  }

 private:

  // helper functions to find a record
  std::tuple<uint32_t, uint32_t> findLeftRecord(uint32_t tid) {

    for (auto &m : leftTIDToRecordMapping) {
      auto it = m.find(tid);
      if(it != m.end()){
        return it->second;
      }
    }

    throw runtime_error("This is bad we did not find a left record");
  }

  // helper functions to find a record
  std::tuple<uint32_t, uint32_t> findRightRecord(uint32_t tid) {

    for (auto &m : rightTIDToRecordMapping) {
      auto it = m.find(tid);
      if(it != m.end()){
        return it->second;
      }
    }

    throw runtime_error("This is bad we did not find a right record");
  }

  // the right tid to record mappings
  std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &leftTIDToRecordMapping;

  // the left tid to record mappings
  std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &rightTIDToRecordMapping;

  // the two columns
  std::vector<Handle < IN1>> * lhsColumn;
  std::vector<Handle < IN2>> * rhsColumn;

  // and the tuple set we return
  TupleSetPtr output;

  // the current record
  uint64_t currentRecord;

  // the start and the end join groups
  uint64_t joinGroupStart;
  uint64_t joinGroupEnd;

  // these are the vectors from where we grab all the tuples
  std::vector<Handle < Vector < std::pair<uint32_t, Handle < IN1>>>>> leftTuples;
  std::vector<Handle < Vector < std::pair<uint32_t, Handle < IN2>>>>> rightTuples;

  PipJoinAggPlanResult::JoinGroups *allJoinGroups;

  // the id of this node
  int32_t nodeID{};

  // the id of the worker
  int32_t workerID{};

  // the number of workers
  int32_t numWorkers{};

  // the plan
  Handle<PipJoinAggPlanResult> plan;

  // the page that contains the plan
  PDBPageHandle planPage;

  // the left page set
  PDBRandomAccessPageSetPtr leftInputPageSet;

  // the right page set
  PDBRandomAccessPageSetPtr rightInputPageSet;

};

}