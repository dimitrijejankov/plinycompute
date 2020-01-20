#pragma once

#include <PDBAnonymousPageSet.h>
#include <JoinTuple.h>
#include <PipJoinAggPlanResult.h>
#include <JoinMap.h>

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

};

// creates the join map
template<typename record_t>
class JoinMapCreator : public JoinMapCreatorBase {
public:

  // we only have binary joins so the tuples are  going to be like this
  using tuple_t = JoinTuple<record_t, char[0]>;

  JoinMapCreator() = default;

  JoinMapCreator(int32_t numThreads,
                 int32_t nodeId,
                 bool isLeft,
                 PDBPageHandle planPage,
                 PDBAnonymousPageSetPtr pageSet,
                 PDBCommunicatorPtr communicator,
                 PDBPageHandle page,
                 PDBLoggerPtr logger) : numThreads(numThreads),
                                        nodeID(nodeId),
                                        isLeft(isLeft),
                                        planPage(std::move(planPage)),
                                        pageSet(std::move(pageSet)),
                                        communicator(std::move(communicator)),
                                        page(std::move(page)),
                                        logger(std::move(logger)) {}


  void run() override {

    // get the
    auto* recordCopy = (Record<PipJoinAggPlanResult>*) this->planPage->getBytes();

    // left and right hashes will be placed here
    std::unordered_map<uint32_t, uint32_t> left;
    std::unordered_map<uint32_t, uint32_t> right;

    // generate the left and right hashes for this node
    (*recordCopy->getRootObject()).generateHashes(nodeID, left, right);

    // depending on the side select the right mapping
    std::unordered_map<uint32_t, uint32_t> *joinMapCreatorHashes;
    if(isLeft) {
      joinMapCreatorHashes = &left;
    }
    else {
      joinMapCreatorHashes = &right;
    }

    // the join map we need to create
    Handle<Vector<Handle<JoinMap<tuple_t>>>> joinMap = nullptr;

    // the page of the page
    PDBPageHandle writePage = nullptr;

    // get the key
    while(true) {

      // get the number of records, if it is -1 we are done here...
      int32_t numRecords = communicator->receivePrimitiveType<int32_t>();
      if(numRecords == -1) {
        break;
      }

      // receive the bytes onto the page
      communicator->receiveBytes(page->getBytes(), error);

      // we start from the beginning
      int currentTuple = 0;

  // jumping here means that we run out of space while we
  // were writing the tuples to the join map
  REDO:

      // if we don't have a page we want to write to get one!
      if(writePage == nullptr) {

        // get a new page to write to
        writePage = pageSet->getNewPage();

        // set it as the current allocation block
        makeObjectAllocatorBlock(writePage->getBytes(), writePage->getSize(), true);
      }

      // if we don't have a join map make it
      if(joinMap == nullptr) {

        // ini ta new join map vector
        joinMap = pdb::makeObject<Vector<Handle<JoinMap<tuple_t>>>>(numThreads, numThreads);
        for(int i = 0; i < numThreads; ++i) {
          (*joinMap)[i] = pdb::makeObject<JoinMap<tuple_t>>();
        }
      }

      // get the records from it
      auto record = ((Record<Vector<std::pair<uint32_t, Handle<record_t>>>> *) page->getBytes());
      auto tuples = record->getRootObject();

      try {

        // insert the page into the join map
        for(; currentTuple < tuples->size(); ++currentTuple) {

          // figure out the aggregation group
          auto tid = (*tuples)[currentTuple].first;
          auto hash = (*joinMapCreatorHashes)[tid];

          // store the tuple
          JoinTuple<record_t, char[0]> &temp = (*joinMap)[hash % numThreads]->push(hash);
          temp.myData = *(*tuples)[currentTuple].second;
        }

      } catch (pdb::NotEnoughSpace &n) {

        // set the root object
        getRecord(joinMap);

        // ok we are done writing to this page
        writePage->unpin();
        writePage = nullptr;
        joinMap = nullptr;

        goto REDO;
      }
    }

    // if we have a write page that means we need to finalize it
    if(writePage != nullptr) {

      // set the root object
      getRecord(joinMap);

      // ok we are done writing to this page and the current iteration
      writePage->unpin();
      writePage = nullptr;
      joinMap = nullptr;
    }

    // this is important it invalidates the current allocation block
    makeObjectAllocatorBlock(1024, true);

    std::cout << "Ended...\n";
  }

  bool getSuccess() override {
    return success;
  }

  const std::string &getError() override {
    return error;
  }

private:

  // the number of threads
  int32_t numThreads{};

  // the id of this node
  int32_t nodeID{};

  // is it the left side or right side
  bool isLeft{};

  // the plan page
  pdb::PDBPageHandle planPage;

  // the page set we are writing to
  PDBAnonymousPageSetPtr pageSet;

  // the communicator we are getting stuff from
  PDBCommunicatorPtr communicator;

  // the page we write stuff to
  PDBPageHandle page;

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