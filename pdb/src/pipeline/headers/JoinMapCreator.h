#pragma once

#include <PDBAnonymousPageSet.h>
#include <JoinTuple.h>
#include <PipJoinAggPlanResult.h>
#include <JoinMap.h>
#include <PDBRandomAccessPageSet.h>

#include <utility>
#include "JoinAggTupleEmitter.h"

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
                 const JoinAggTupleEmitterPtr& emitter,
                 bool isLHS,
                 PDBLoggerPtr logger) : pageSet(std::move(pageSet)),
                                        communicator(std::move(communicator)),
                                        logger(std::move(logger)),
                                        emitter(std::dynamic_pointer_cast<JoinAggTupleEmitter>(emitter)),
                                        joinSide(isLHS ? LHS : RHS) {}


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
      auto record = ((Record<Vector<std::pair<uint32_t, Handle<Nothing>>>> *) recordPage->getBytes());
      auto tuples = record->getRootObject();

      // freeze it
      recordPage->freezeSize(record->numBytes());

      // forward the stuff to the emitter
      if(joinSide == LHS) {
        emitter->gotLHS(tuples, pageIndex);
      }
      else {
        emitter->gotRHS(tuples, pageIndex);
      }
    }
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

  // what side of the join is this
  enum {LHS, RHS} joinSide;

  // the emitter
  std::shared_ptr<JoinAggTupleEmitter> emitter;

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