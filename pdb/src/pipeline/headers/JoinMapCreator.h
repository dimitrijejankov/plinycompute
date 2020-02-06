#pragma once

#include <PDBAnonymousPageSet.h>
#include <JoinTuple.h>
#include <PipJoinAggPlanResult.h>
#include <JoinMap.h>
#include <PDBRandomAccessPageSet.h>

namespace pdb {

// the base class so we can run it
class JoinMapCreatorBase {
public:

  // runs the map creator
  virtual void run() = 0;

  // check if we succeeded
  virtual bool getSuccess() = 0;

  // check if we failed
  virtual const std::string &getError() = 0;

  // extracts the TID map
  virtual std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>> extractTIDMap()  = 0;

};

// creates the join map
template<typename record_t>
class JoinMapCreator : public JoinMapCreatorBase {
public:

  JoinMapCreator() = default;

  JoinMapCreator(PDBRandomAccessPageSetPtr pageSet,
                 PDBCommunicatorPtr communicator,
                 PDBLoggerPtr logger) : pageSet(std::move(pageSet)),
                                        communicator(std::move(communicator)),
                                        logger(std::move(logger)) {}


  void run() override {

    // get the key
    while(true) {

      // get the number of records, if it is -1 we are done here...
      int32_t numRecords = communicator->receivePrimitiveType<int32_t>();
      if(numRecords == -1) {
        break;
      }

      // get a new page
      auto [pageIndex, recordPage] = pageSet->getNextPageWithIndex();

      // receive the bytes onto the page
      communicator->receiveBytes(recordPage->getBytes(), error);

      // get the records from it
      auto record = ((Record<Vector<std::pair<uint32_t, Handle<record_t>>>> *) recordPage->getBytes());
      auto tuples = record->getRootObject();

      // insert the page into the join map
      for(uint32_t currentTuple = 0; currentTuple < tuples->size(); ++currentTuple) {

        // figure out the aggregation group
        auto tid = (*tuples)[currentTuple].first;

        // insert into the mapping
        tidToRecordMapping.insert(std::pair(tid, std::make_tuple(pageIndex, currentTuple)));
      }

      // unpin the record page
      recordPage->unpin();
    }

    std::cout << "JoinMapCreator Finished!\n";
  }

  bool getSuccess() override {
    return success;
  }

  const std::string &getError() override {
    return error;
  }

  std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>> extractTIDMap() override {
    return std::move(tidToRecordMapping);
  }

private:

  // we are mapping the tid to the
  std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>> tidToRecordMapping;

  // the page set we are writing to
  PDBRandomAccessPageSetPtr pageSet;

  // the communicator we are getting stuff from
  PDBCommunicatorPtr communicator;

  // the logger
  PDBLoggerPtr logger;

  // did we succeed
  bool success = true;

  // was there an error
  std::string error;
};

// make the shared ptr for this
using JoinMapCreatorPtr = std::shared_ptr<JoinMapCreatorBase>;
}