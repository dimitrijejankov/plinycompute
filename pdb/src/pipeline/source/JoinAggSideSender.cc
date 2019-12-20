#include <JoinAggSideSender.h>
#include <condition_variable>

int32_t pdb::JoinAggSideSender::queueToSend(std::vector<std::pair<uint32_t, pdb::Handle<pdb::matrix::MatrixBlock>>> *records) {

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
  toSend.push({currID, records});

  // queue the identifier
  queuedIdentifiers.insert(currID);

  // we added stuff to the queue
  cv.notify_all();

  // store it into the queue
  return currID;
}

void pdb::JoinAggSideSender::waitToFinish(int32_t id) {

  // we don't wait for -1 since that means the vector was empty
  if(id == -1) {
    return;
  }

  // lock the structure
  std::unique_lock<std::mutex> lk(m);

  // wait till the sending thread is done with the stuff we were sending
  cv.wait(lk, [&]{ return  queuedIdentifiers.find(id) == queuedIdentifiers.end();});
}

void pdb::JoinAggSideSender::run() {

  // the error is stored here if any
  std::string error;

  // lock the structure
  std::unique_lock<std::mutex> lk(m);

  // do the sending
  while(true) {

    // wait till we have something to send, or we are done sending
    cv.wait(lk, [&]{ return !toSend.empty() || !stillSending; });

    std::cout << "Sending...\n";
    // break if we are done sending
    if(!stillSending && toSend.empty()) {
      break;
    }

    // grab it from the queue
    auto recs = toSend.front();
    toSend.pop();

    // use the page
    const UseTemporaryAllocationBlock tempBlock{page->getBytes(), page->getSize()};

REDO:
    // empty out the page
    getAllocator().emptyOutBlock(page->getBytes());

    // create a vector to store stuff
    Handle<Vector<std::pair<uint32_t, recordHandle>>> vec = pdb::makeObject<Vector<std::pair<uint32_t, recordHandle>>>();

    // how many did we put on a page
    int32_t numOnPage = 0;

    // go through the records and put them on a page
    for(int i = 0; i < recs.second->size(); ++i) {

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
        getRecord(vec);

        // send the number of objects
        if(!communicator->sendPrimitiveType(numOnPage)) {
          return;
        }

        // send the bytes
        if(!communicator->sendBytes(page->getBytes(), page->getSize(), error)) {

          // finish here we got an error
          return;
        }

        // redo since we have stuff left
        goto REDO;
      }
    }

    // if there are records on the page send them
    if(numOnPage != 0) {

      // set the root object
      getRecord(vec);

      // send the number of objects
      if(!communicator->sendPrimitiveType(numOnPage)) {
        return;
      }

      // send the bytes
      if(!communicator->sendBytes(page->getBytes(), page->getSize(), error)) {

        // finish here we got an error
        return;
      }

      std::cout << "Sent... \n";
    }

    // unqueue the identifier
    queuedIdentifiers.erase(recs.first);
    cv.notify_all();
  }

  // that we are done
  std::cout << "Finished... \n";
  if(!communicator->sendPrimitiveType(-1)) {
    return;
  }
}

void pdb::JoinAggSideSender::shutdown() {

  // lock the structure
  std::unique_lock<std::mutex> lk(m);

  // mark that we are not sending
  stillSending = false;

  // notify that we are done
  cv.notify_all();
}
