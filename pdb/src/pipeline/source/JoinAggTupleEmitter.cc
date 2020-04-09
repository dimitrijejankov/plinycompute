#include <JoinAggTupleEmitter.h>
#include <PDBPageHandle.h>
#include <Record.h>

#include <utility>

namespace pdb {

pdb::JoinAggTupleEmitter::JoinAggTupleEmitter(PDBPageHandle planPage, int numThreads, int32_t nodeID) : planPage(std::move(planPage)),
                                                                                                        numThreads(numThreads),
                                                                                                        threadsWaiting(numThreads),
                                                                                                        hasEnded(false) {

  // get the plan result so we can do stuff
  this->planPage->repin();
  auto *recordCopy = (Record<PipJoinAggPlanResult> *) this->planPage->getBytes();
  plan = recordCopy->getRootObject();

  // init the structure that keeps track on what thread we assigned an aggregation group
  threadAssigned.resize(plan->numAggGroups);
  for(auto &g : threadAssigned) { g = -1; }

  // the join groups
  auto myJoinGroups = &((*plan->joinGroupsPerNode)[nodeID]);

  // reserve the size the records
  recordsToEmit.reserve(myJoinGroups->size());

  // go through the groups
  for(int i = 0; i < myJoinGroups->size(); ++i) {

    // get the join group
    auto &jg = (*myJoinGroups)[i];

    // store the mappings
    lhsMappings.insert({jg.lhsTID, i});
    rhsMappings.insert({jg.rhsTID, i});

    // add the record
    recordsToEmit.emplace_back(jg.aggTID, i);
  }

  // init the booleans we use for waiting
  for(auto &t : threadsWaiting) {
    t.buffer.reserve(1000);
  }
}

void JoinAggTupleEmitter::gotLHS(const Handle<Vector<std::pair<uint32_t, Handle<Nothing>>>> &lhs, int32_t pageIndex) {

  // lock the lhs
  std::unique_lock<std::mutex> lck(m);

  // go through the lhs
  for(int i = 0; i < lhs->size(); ++i) {

    // find all join records with this lhs record
    auto range = lhsMappings.equal_range((*lhs)[i].first);

    // go through all join records that have a matching tid
    for(auto it = range.first; it != range.second; ++it) {

      // get the reference to the record
      auto &record = recordsToEmit[it->second];

      // update the info
      record.lhs_record = i;
      record.lhs_page = pageIndex;

      // emit the record
      if(record.rhs_record != -1) {

        // check if we what an assignment
        if(threadAssigned[record.agg_group] == -1) {

          // assign the thread
          threadAssigned[record.agg_group] = nextThread;

          // figure out the next thread
          nextThread = (nextThread + 1) % numThreads;
        }

        // figure out where it is assigned
        auto assignedThread = threadAssigned[record.agg_group];

        // set the last page
        threadsWaiting[assignedThread].lastLHSPage = std::max(threadsWaiting[assignedThread].lastLHSPage, record.lhs_page + 1);
        threadsWaiting[assignedThread].lastRHSPage = std::max(threadsWaiting[assignedThread].lastRHSPage, record.rhs_page + 1);

        // emit the record
        numEms++;
        threadsWaiting[assignedThread].buffer.emplace_back(record);
      }
    }
  }
}

void JoinAggTupleEmitter::gotRHS(const Handle<Vector<std::pair<uint32_t, Handle<Nothing>>>> &rhs, int32_t pageIndex) {

  // lock the lhs
  std::unique_lock<std::mutex> lck(m);

  // go through the rhs
  for(int i = 0; i < rhs->size(); ++i) {

    // find all join records with this lhs record
    auto range = rhsMappings.equal_range((*rhs)[i].first);

    // go through all join records that have a matching tid
    for(auto it = range.first; it != range.second; ++it){

      // get the reference to the record
      auto &record = recordsToEmit[it->second];

      // update the info
      record.rhs_record = i;
      record.rhs_page = pageIndex;

      // emit the record
      if(record.lhs_record != -1) {

        // check if we what an assignment
        if(threadAssigned[record.agg_group] == -1) {

          // assign the thread
          threadAssigned[record.agg_group] = nextThread;

          // figure out the next thread
          nextThread = (nextThread + 1) % numThreads;
        }

        // figure out where it is assigned
        auto assignedThread = threadAssigned[record.agg_group];

        // set the last page
        threadsWaiting[assignedThread].lastRHSPage = std::max(threadsWaiting[assignedThread].lastRHSPage, record.rhs_page + 1);
        threadsWaiting[assignedThread].lastLHSPage = std::max(threadsWaiting[assignedThread].lastLHSPage, record.lhs_page + 1);

        // emit the record
        numEms++;
        threadsWaiting[assignedThread].buffer.emplace_back(record);
      }
    }
  }

  // go through the threads that are waiting
  for(auto &t : threadsWaiting) {

    // if we have records buffered
    if(!t.buffer.empty()) {

      // mark that we got records
      t.gotRecords = true;
      t.cv.notify_one();
    }
  }
}

void JoinAggTupleEmitter::printEms() {
  std::cout << "Emitted : " << numEms << '\n';
}

void JoinAggTupleEmitter::end() {

  // lock this and add records
  std::unique_lock<std::mutex> lck(m);

  // mark that it has ended
  hasEnded = true;

  // go through the threads that are waiting
  for(auto &t : threadsWaiting) {
    t.cv.notify_one();
  }
}

void JoinAggTupleEmitter::getRecords(std::vector<JoinedRecord> &putHere, int32_t &lastLHSPage, int32_t &lastRHSPage, int32_t threadID) {

  // clear the vector
  putHere.clear();

  {

    // lock this and add records
    std::unique_lock<std::mutex> lck(m);

    // reset the value so emitter knows we need to wait
    if(threadsWaiting[threadID].buffer.empty()) {
      threadsWaiting[threadID].gotRecords = false;
    }

    // wait till we get some records
    while (!threadsWaiting[threadID].gotRecords && !hasEnded) {
      threadsWaiting[threadID].cv.wait(lck);
    }

    // swap the threads
    std::swap(putHere, threadsWaiting[threadID].buffer);

    // set the last page we need to get
    lastLHSPage = threadsWaiting[threadID].lastLHSPage;
    lastRHSPage = threadsWaiting[threadID].lastRHSPage;
  }

  // sort the records so that the go in a nice order
  std::sort(putHere.begin(), putHere.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.group_id < rhs.group_id;
  });
}


}
