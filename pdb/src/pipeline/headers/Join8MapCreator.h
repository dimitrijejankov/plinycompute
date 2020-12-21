#pragma once

#include <PDBAnonymousPageSet.h>
#include <JoinTuple.h>
#include <PipJoinAggPlanResult.h>
#include <JoinMap.h>
#include <PDBRandomAccessPageSet.h>

namespace pdb {

// creates the join map
class Join8MapCreator {
 public:

  Join8MapCreator() = default;

  Join8MapCreator(PDBRandomAccessPageSetPtr pageSet,
                  PDBCommunicatorPtr communicator,
                  PDBLoggerPtr logger,
                  PDBPageHandle page) : pageSet(std::move(pageSet)),
                                        planPage(std::move(page)),
                                        communicator(std::move(communicator)),
                                        logger(std::move(logger)) {}

  void run() {

    // get plan from the page
    auto* rr = (Record<JoinPlannerResult>*) planPage->getBytes();
    auto planResult = rr->getRootObject();

    // get the key
    while (true) {

      // get the number of records, if it is -1 we are done here...
      int32_t numRecords = communicator->receivePrimitiveType<int32_t>();
      if (numRecords == -1) {
        break;
      }

      // get a new page
      auto[pageIndex, recordPage] = pageSet->getNextPageWithIndex();

      // receive the bytes onto the page
      communicator->receiveBytes(recordPage->getBytes(), error);

      // get the records from it
      auto record = ((Record<Vector<Handle<TRABlock>>> *) recordPage->getBytes());
      auto tuples = record->getRootObject();

      // freeze it
      recordPage->freezeSize(record->numBytes());

      // insert the page into the join map
      for (uint32_t currentTuple = 0; currentTuple < tuples->size(); ++currentTuple) {

        // figure out the aggregation group
        auto r = (*tuples)[currentTuple];

        // get the tid
        auto tid = (*planResult->records0)[*r->getKey()];

        // insert into the join_group_mapping
        tidToRecordMapping.insert(std::pair(tid, std::make_tuple(pageIndex, currentTuple)));
      }

      // unpin the record page
      recordPage->unpin();
    }
  }

  bool getSuccess() {
    return success;
  }

  const std::string &getError() {
    return error;
  }

  std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>> extractTIDMap() {
    return std::move(tidToRecordMapping);
  }

 private:

  // we are join_group_mapping the tid to the
  std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>> tidToRecordMapping;

  // the page set we are writing to
  PDBRandomAccessPageSetPtr pageSet;

  // the communicator we are getting stuff from
  PDBCommunicatorPtr communicator;

  // the page where we keep the plan
  PDBPageHandle planPage;

  // the logger
  PDBLoggerPtr logger;

  // did we succeed
  bool success = true;

  // was there an error
  std::string error;
};

// make the shared ptr for this
using Join8MapCreatorPtr = std::shared_ptr<Join8MapCreator>;
}