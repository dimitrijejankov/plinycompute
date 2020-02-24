#pragma once

#include <utility>
#include <PDBCommunicator.h>
#include <PDBPageHandle.h>
#include <condition_variable>
#include <unordered_set>
#include <queue>

namespace pdb {

// just the base class
class JoinAggSideSenderBase {
public:

  // runs the sender
  virtual void run() = 0;

  // shuts the sender down
  virtual void shutdown() = 0;

  // queues a vector of records to send the vector is going to be casted
  virtual int32_t queueToSend(void* records) = 0;

  // waits for the sending to finish
  virtual void waitToFinish(int32_t id) = 0;
};

template<typename record_t>
class JoinAggSideSender : public JoinAggSideSenderBase {
public:

  // the record handle
  using recordHandle = pdb::Handle<record_t>;

  JoinAggSideSender(PDBPageHandle page, PDBCommunicatorPtr  comm) : page(std::move(page)),
                                                                    communicator(std::move(comm)) {}

  /**
   * Takes in the records and sends them to a particular node
   * @param records - the records we want to send
   * @return the identifier which is passed to waitToFinish
   */
  int32_t queueToSend(void* records) override {
    return _queueToSend((std::vector<std::pair<uint32_t, recordHandle>> *) records);
  }

  /**
   * This is the actual implementation
   * @param records - the records we want to send
   * @return the identifier which is passed to waitToFinish
   */
  int32_t _queueToSend(std::vector<std::pair<uint32_t, recordHandle>> *records) {

    // don't have anything to send
    if(records->empty()) {
      return -1;
    }

    // lock the page structure
    unique_lock<mutex> lck(m);

    // figure out the current id
    auto currID = nextID;

    // figure out the next and make sure there is no overflow...
    nextID = nextID + 1 < 0 ? 0 : nextID + 1;

    // store the records
    toSend.push_back({currID, records});

    // queue the identifier
    queuedIdentifiers.insert(currID);

    // we added stuff to the queue
    cv.notify_all();

    // store it into the queue
    return currID;
  }

  /**
   * Wait to finish sending the records
   * @param id - the id of the records we are waiting
   */
  void waitToFinish(int32_t id) override {

    // we don't wait for -1 since that means the vector was empty
    if(id == -1) {
      return;
    }

    // lock the structure
    std::unique_lock<std::mutex> lk(m);

    // wait till the sending thread is done with the stuff we were sending
    cv.wait(lk, [&]{ return  queuedIdentifiers.find(id) == queuedIdentifiers.end();});
  }

  /**
   * run the sender
   */
  void run() override {

    // lock the structure
    std::unique_lock<std::mutex> lk(m);

REDO:

    // how many did we put on a page
    int32_t numOnPage = 0;

    // use the page
    const UseTemporaryAllocationBlock tempBlock{page->getBytes(), page->getSize()};

    // empty out the page
    getAllocator().emptyOutBlock(page->getBytes());

    // create a vector to store stuff
    Handle<Vector<std::pair<uint32_t, recordHandle>>> vec = pdb::makeObject<Vector<std::pair<uint32_t, recordHandle>>>();

    // do the sending
    while(true) {

      // wait till we have something to send, or we are done sending
      cv.wait(lk, [&]{ return !toSend.empty() || !stillSending; });

      // break if we are done sending
      if(!stillSending && toSend.empty()) {
        break;
      }

      // grab it from the queue
      auto recs = toSend.front();
      toSend.pop_front();

      // go through the records and put them on a page
      size_t numRec = recs.second->size();
      for(int i = 0; i < numRec; ++i) {

        // try to insert the record
        try {

          // insert the record
          vec->push_back(recs.second->back());

          // the record is on the page
          numOnPage++;

          // remove the last one since we manage to add it to the page
          recs.second->pop_back();

        } catch (pdb::NotEnoughSpace &n) {

          // set the root object
          auto record = getRecord(vec);

          // send the number of objects
          if(!communicator->sendPrimitiveType(numOnPage)) {
            return;
          }

          // send the bytes
          if(!communicator->sendBytes(page->getBytes(), record->numBytes(), error)) {

            // finish here we got an error
            return;
          }

          // move them to the frond
          toSend.push_front(recs);

          // empty out the page
          getAllocator().emptyOutBlock(page->getBytes());

          // redo since we have stuff left
          goto REDO;
        }
      }

      // unqueue the identifier
      queuedIdentifiers.erase(recs.first);
      cv.notify_all();

      // get how much there is left till we reach the end of the page
      auto freePercentage = (100 * getAllocator().getFreeBytesAtTheEnd()) / page->getSize();

      // if there are records on the page and we are over 90% (less than 10% free) send the page
      // or if we are done sending and this is the last batch just send them over
      if(numOnPage != 0 && freePercentage <= 10) {

        // set the root object
        auto record = getRecord(vec);

        // send the number of objects
        if(!communicator->sendPrimitiveType(numOnPage)) {
          return;
        }

        // send the bytes
        if(!communicator->sendBytes(page->getBytes(), record->numBytes(), error)) {

          // finish here we got an error
          return;
        }

        // empty out the page
        getAllocator().emptyOutBlock(page->getBytes());

        // go to redo since we need a new page
        goto REDO;
      }

    }

    // if something is left send it
    if(numOnPage != 0) {

      // set the root object
      auto record = getRecord(vec);

      // send the number of objects
      if(!communicator->sendPrimitiveType(numOnPage)) {
        return;
      }

      // send the bytes
      if(!communicator->sendBytes(page->getBytes(), record->numBytes(), error)) {

        // finish here we got an error
        return;
      }

      // we just sent the page empty it out
      getAllocator().emptyOutBlock(page->getBytes());
    }

    // that we are done
    if(!communicator->sendPrimitiveType(-1)) {
      return;
    }
  }

  /**
   * shutdown the sender
   */
  void shutdown() override {

    // lock the structure
    std::unique_lock<std::mutex> lk(m);

    // mark that we are not sending
    stillSending = false;

    // notify that we are done
    cv.notify_all();
  }


 private:

  /**
   * the error is stored here if any
   */
  std::string error;

  /**
   * This indicates whether we are sending it
   */
  bool stillSending = true;

  /**
   * the queue of record vectors with their identifiers
   */
  std::deque<std::pair<int32_t, std::vector<std::pair<uint32_t, recordHandle>>*>> toSend;

  /**
   * the identifiers that are currently queued
   */
   std::unordered_set<int32_t> queuedIdentifiers;

   /**
    * the next id we are going to assign
    */
   int32_t nextID = 0;

  /**
   * The mutex to sync the sending
   */
  std::mutex m;

  /**
   * The conditional variable for waiting
   */
  std::condition_variable cv;

  /**
   * This is the page we are putting the stuff we want to send to
   */
   PDBPageHandle page;

  /**
   * This thing is sending our stuff to the right node
   */
  PDBCommunicatorPtr communicator;

};

// make a shared ptr shortcut
using JoinAggSideSenderPtr = std::shared_ptr<JoinAggSideSenderBase>;

}