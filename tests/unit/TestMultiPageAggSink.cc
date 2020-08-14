#include <iostream>
#include <vector>
#include <list>
#include <random>
#include <ctime>
#include <chrono>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <SingleNodePlanner.h>
#include <PDBBufferManagerImpl.h>
#include "../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"
#include "PDBMap.h"

#include <condition_variable>
#include <mutex>
#include <GenericWork.h>
#include <PageProcessor.h>

using OutputType = pdb::matrix::MatrixBlock;
using KeyType = pdb::matrix::MatrixBlockMeta;
using ValueType = pdb::matrix::MatrixBlockData;

const int32_t numPagesToAggregate = 20;

class record_forwarding_queue_t {
public:

  void wait_for_records(std::shared_ptr<std::list<std::pair<void*, void*>>> &records) {

    // lock the queue
    std::unique_lock<std::mutex> lock(m);

    // wait to get some records or that we are finished
    cv.wait(lock, [&] { return finished || _records != nullptr; });

    // give it the records
    records = _records;
  }

  void wait_till_processed() {

    // lock the queue
    std::unique_lock<std::mutex> lock(m);

    // wait to get some records or that we are finished
    cv.wait(lock, [&] { return _records == nullptr; });
  }

  void processed() {

    // lock the queue
    std::unique_lock<std::mutex> lock(m);

    // invalidate the records
    _records = nullptr;

    // notify all the waiters
    cv.notify_all();
  }

  void enqueue(const std::shared_ptr<std::list<std::pair<void*, void*>>> &records) {

    // lock the queue
    std::unique_lock<std::mutex> lock(m);

    // mark as finished
    if(records == nullptr) {
      finished = true;
    }

    // set the records
    _records = records;

    // notify all the waiters
    cv.notify_all();
  }

private:

  // did we finish this
  bool finished{false};

  // we use this for synchronization
  std::mutex m;
  std::condition_variable cv;

  // the records we need to give
  std::shared_ptr<std::list<std::pair<void*, void*>>> _records = nullptr;
};

void processing_thread(const pdb::PDBBufferManagerInterfacePtr &mgr,
                       record_forwarding_queue_t &records_queue,
                       pdb::PDBWorkerQueuePtr workerQueue) {

  // get the page
  auto outputPage = mgr->getPage();

  // filled up the threshold tells when to stop adding new keys, currently set to 15%
  auto fill_threshold = (size_t) (outputPage->getSize() * 0.15);

  // make the allocation block
  pdb::makeObjectAllocatorBlock(outputPage->getBytes(), outputPage->getSize(), true);

  // make the map
  pdb::Handle <pdb::Map <KeyType, ValueType>> myMap = pdb::makeObject <pdb::Map <KeyType, ValueType>> ();

  // this is used if we spawned a child
  bool hasChild = false;
  record_forwarding_queue_t child_queue;

  // the buzzer we use for the child
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {});

  // while we have something do stuff
  while (true) {

    // grab the next batch of records from this
    std::shared_ptr<std::list<std::pair<void*, void*>>> records;
    records_queue.wait_for_records(records);

    // break out of this loop since we are done with pages
    if(records == nullptr) {

      // if it has a child
      if (hasChild) {

        // insert this into the child queue
        child_queue.enqueue(nullptr);

        // wait till processed
        child_queue.wait_till_processed();
      }

      // mark as processed
      child_queue.processed();

      // break out of this loop
      break;
    }

    try {

      // go through all the records
      for(auto it = records->begin(); it != records->end();) {

        // are we aggregating new that means we can still do more
        if(!hasChild) {

          // get the key and value
          auto &key = *((KeyType*) it->first);
          auto &value = *((ValueType*) it->second);

          if (myMap->count(key) == 0) {

            // copy and add to hash map
            ValueType &temp = (*myMap)[key];
            temp = value;

          }
          else {

            // copy and add to hash map
            ValueType &temp = (*myMap)[key];
            ValueType copy = temp;
            temp = copy + value;
          }

          // if there is less space left than the threshold we stop filling
          if(fill_threshold >= pdb::getAllocator().getFreeBytesAtTheEnd()) {

            // mark that we are not aggregating
            hasChild = true;

            // get a worker from the server
            pdb::PDBWorkerPtr worker = workerQueue->getWorker();

            // start a child
            pdb::PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&](const PDBBuzzerPtr& callerBuzzer) {
              processing_thread(mgr, child_queue, workerQueue);
            });

            // run the work
            worker->execute(myWork, tempBuzzer);
          }

          // remove it
          it = records->erase(it);
        }
        else {

          // get the key and value
          auto &key = *((KeyType*) it->first);
          auto &value = *((ValueType*) it->second);

          // try to find the key
          if (myMap->count(key) != 0) {

            // copy and add to hash map
            ValueType &temp = (*myMap)[key];
            ValueType copy = temp;
            temp = copy + value;

            // remove it
            it = records->erase(it);
          }
          else {

            // just skip it
            it++;
          }
        }
      }

    } catch (pdb::NotEnoughSpace &n) {

      // we do not deal with this, it must fit into a single hash table
      throw n;
    }


    // if it has a child give it some stuff to do
    if (hasChild) {

      // forward the records to the children
      child_queue.enqueue(records);

      // wait till processed
      child_queue.wait_till_processed();
    }

    // mark as processed
    records_queue.processed();
  }

  // make this the root object of the allocation block
  pdb::getRecord(myMap);

  // TODO make this nicer (invalidates the allocation block)
  pdb::makeObjectAllocatorBlock(1024, true);
}

void process_map(pdb::PDBBufferManagerInterfacePtr &mgr, pdb::PDBWorkerQueuePtr &workers, std::vector<pdb::PDBPageHandle> &pages) {

  record_forwarding_queue_t child_queue;

  // the buzzer we use for the child
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {});

  // get a worker from the server
  pdb::PDBWorkerPtr worker = workers->getWorker();

  // start a child
  pdb::PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&](const PDBBuzzerPtr& callerBuzzer) {
    processing_thread(mgr, child_queue, workers);
  });

  // run the work
  worker->execute(myWork, tempBuzzer);

  while(!pages.empty()) {

    // get a page
    auto page = pages.back();
    pages.pop_back();

    // repin the page
    page->repin();

    // get the records from it
    auto record = ((pdb::Record<pdb::Map <KeyType, ValueType>> *) page->getBytes());
    auto tuples = record->getRootObject();

    // go through each key, value pair in the hash map we want to merge
    std::shared_ptr<std::list<std::pair<void*, void*>>> records = std::make_shared<std::list<std::pair<void*, void*>>>();
    for(auto it = tuples->begin(); it != tuples->end(); ++it) {
      records->emplace_back(&(*it).key, &(*it).value);
    }

    // process
    child_queue.enqueue(records);

    // wait till everything is processed
    child_queue.wait_till_processed();

    // unpin the page
    page->unpin();
  }

  // mark that we are done
  child_queue.enqueue(nullptr);

  // wait till we have processed everything
  child_queue.wait_till_processed();
}

int main() {

  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // make the block we use them
  pdb::Handle<pdb::matrix::MatrixBlock> tmp = pdb::makeObject <pdb::matrix::MatrixBlock> (0,0, 100, 100);

  // make a worker queue
  auto workers = make_shared<pdb::PDBWorkerQueue>(make_shared<pdb::PDBLogger>("worker.log"), 10);

  // create the buffer manager
  std::shared_ptr<pdb::PDBBufferManagerImpl> mgr = std::make_shared<pdb::PDBBufferManagerImpl>();
  mgr->initialize("tempDSFSD",
                  1024u * 1024u,
                  16,
                  "metadata",
                  ".");


  // filled up the threshold tells when to stop adding new keys, currently set to 15%
  auto fill_threshold = (size_t) (mgr->getMaxPageSize() * 0.15);

  std::vector<pdb::PDBPageHandle> pages;
  for(int i = 0; i < numPagesToAggregate; ++i) {

    // get a new page
    pages.push_back(mgr->getPage());

    // make this the new allocator
    pdb::makeObjectAllocatorBlock(pages.back()->getBytes(), pages.back()->getSize(), true);

    // make the map
    pdb::Handle <pdb::Map <KeyType, ValueType>> returnVal = pdb::makeObject <pdb::Map <KeyType, ValueType>> ();

    //
    int32_t idx = i % 5;
    while(true) {

      // set the next row id
      tmp->metaData->rowID = idx++;

      // init everything to idx
      for(int r = 0; r < 100 * 100; r++) { tmp->data->data->c_ptr()[r] = 1.0f * idx; }

      // copy this to the hash map
      ValueType &temp = (*returnVal)[tmp->getKeyRef()];
      temp = tmp->getValueRef();

      // break out if we have filled the page to 85%
      if(fill_threshold >= pdb::getAllocator().getFreeBytesAtTheEnd()) {
        break;
      }
    }

    pdb::getRecord(returnVal);

    // unpin the page
    pages.back()->unpin();
  }

  ////

  // make the work
  process_map(mgr, workers, pages);
}