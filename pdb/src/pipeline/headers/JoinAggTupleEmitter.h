#pragma once

#include <PDBPageHandle.h>
#include <vector>
#include <unordered_map>
#include <PipJoinAggPlanResult.h>
#include <Handle.h>
#include <condition_variable>

namespace pdb {

class JoinAggTupleEmitter {
public:

  explicit JoinAggTupleEmitter(PDBPageHandle planPage, int numThreads, int32_t nodeID);

  struct JoinedRecord {

    explicit JoinedRecord(int32_t agg_group, int32_t group_id) : agg_group(agg_group), group_id(group_id) {}

    // tells us where the lhs record is located
    int32_t lhs_page = -1;
    int32_t lhs_record = -1;

    // tells us where the rhs record is located
    int32_t rhs_page = -1;
    int32_t rhs_record = -1;

    // the the group it belongs to
    int32_t agg_group = -1;
    int32_t group_id = -1;
  };

  struct ThreadInfo {
    std::mutex m;
    std::condition_variable cv;
    bool gotRecords{false};
    std::vector<JoinedRecord> buffer;
    int32_t lastLHSPage = 0;
    int32_t lastRHSPage = 0;
  };

  // got the lhs records
  void gotLHS(const Handle<Vector<std::pair<uint32_t, Handle<Nothing>>>> &lhs, int32_t pageIndex);

  // got the rhs records
  void gotRHS(const Handle<Vector<std::pair<uint32_t, Handle<Nothing>>>> &rhs, int32_t pageIndex);

  void getRecords(std::vector<JoinedRecord> &putHere, int32_t &lastLHSPage, int32_t &lastRHSPage, int32_t threadID);

  // print emit stats
  void printEms();

  //
  void end();

  // this locks this
  std::mutex m;

  // next

  // all the threads that are waiting for records to join
  std::vector<ThreadInfo> threadsWaiting;

  // the plan page
  PDBPageHandle planPage{};

  Handle<PipJoinAggPlanResult> plan;

  // the records we want to emit
  std::vector<JoinedRecord> recordsToEmit;

  // the next thread
  atomic_int32_t nextThread = 0;

  // number of threads
  int32_t numThreads = 0;

  //
  int32_t numEms = 0;

  bool hasEnded = false;

  // the mappings for the left and right records
  std::unordered_multimap<uint32_t, int32_t> lhsMappings;
  std::unordered_multimap<uint32_t, int32_t> rhsMappings;

  // the threads assigned
  std::vector<int8_t> threadAssigned;
};

using JoinAggTupleEmitterPtr = std::shared_ptr<JoinAggTupleEmitter>;

}