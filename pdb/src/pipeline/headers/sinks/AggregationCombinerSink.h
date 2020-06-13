//
// Created by dimitrije on 3/27/19.
//

#ifndef PDB_AGGREGATIONCOMBINERSINK_H
#define PDB_AGGREGATIONCOMBINERSINK_H

#include <ComputeSink.h>
#include <stdexcept>
#include <PDBPageHandle.h>
#include <list>
#include <GenericWork.h>
#include <PDBAnonymousPageSet.h>

namespace pdb {

class AggregationCombinerSinkBase : public ComputeSink {
 public:

  virtual std::shared_ptr<std::list<std::pair<void*, void*>>> getTupleList(const pdb::PDBPageHandle &page) = 0;

  // we use this to forward the page between the workers
  class record_forwarding_queue_t {
   public:

    void wait_for_records(std::shared_ptr<std::list<std::pair<void*, void*>>> &records) {

      // lock the buffer manager
      std::unique_lock<std::mutex> lock(m);

      // wait to get some records or that we are finished
      cv.wait(lock, [&] { return state == state_t::DONE || _records != nullptr; });

      // give it the records
      records = _records;
    }

    void wait_till_processed() {

      // lock the buffer manager
      std::unique_lock<std::mutex> lock(m);

      // wait to get some records or that we are finished
      cv.wait(lock, [&] { return (_records == nullptr && state == state_t::RUNNING) || state == state_t::SHUT_DOWN; });
    }

    void processed() {

      // lock the buffer manager
      std::unique_lock<std::mutex> lock(m);

      // if we are finished and processed set that the thread is shutdown
      if(state == state_t::DONE) {
        state = state_t::SHUT_DOWN;
      }

      // invalidate the records
      _records = nullptr;

      // notify all the waiters
      cv.notify_all();
    }

    void enqueue(const std::shared_ptr<std::list<std::pair<void*, void*>>> &records) {

      // lock the buffer manager
      std::unique_lock<std::mutex> lock(m);

      // mark as finished
      if(records == nullptr) {
        state = state_t::DONE;
      }

      // set the records
      _records = records;

      // notify all the waiters
      cv.notify_all();
    }

private:

    enum class state_t {
      RUNNING,
      DONE,
      SHUT_DOWN
    };

    // the state the queue is in
    state_t state{state_t::RUNNING};

    // we use this for synchronization
    std::mutex m;
    std::condition_variable cv;

    // the records we need to give
    std::shared_ptr<std::list<std::pair<void*, void*>>> _records = nullptr;
  };

  // this runs the processing thread
  virtual void processing_thread(const pdb::PDBAnonymousPageSetPtr &mgr,
                                 record_forwarding_queue_t &records_queue,
                                 pdb::PDBWorkerQueuePtr workerQueue) = 0;

};

using AggregationCombinerSinkBasePtr = std::shared_ptr<AggregationCombinerSinkBase>;

template<class KeyType, class ValueType>
class AggregationCombinerSink : public AggregationCombinerSinkBase {
 public:

  explicit AggregationCombinerSink(size_t workerID) : workerID(workerID) {

    // set the count to zero
    counts = 0;
  }

  Handle<Object> createNewOutputContainer() override {}

  // throws an exception
  void writeOut(TupleSetPtr writeMe, Handle<Object> &writeToMe) override {
    throw std::runtime_error("AggregationCombinerSink can not write out tuple sets only pages.");
  }

  // throws an exception
  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override {
    throw std::runtime_error("AggregationCombinerSink can not write out tuple sets only pages.");
  }

  // returns the number of records in the aggregation sink
  uint64_t getNumRecords(Handle<Object> &writeToMe) override {
    return counts;
  }

  void processing_thread(const pdb::PDBAnonymousPageSetPtr &mgr,
                         record_forwarding_queue_t &records_queue,
                         pdb::PDBWorkerQueuePtr workerQueue) override {
    // get the page
    auto outputPage = mgr->getNewPage();

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
        records_queue.processed();

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

              // increment the counts
              counts++;
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

              // increment the counts
              counts++;
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

  // return the list of all the tuples as void* that need to be aggregated
  std::shared_ptr<std::list<std::pair<void*, void*>>> getTupleList(const pdb::PDBPageHandle &page) override {

    // get the records from it
    // grab the hash table
    Handle<Object> hashTable = ((Record<Object> *) page->getBytes())->getRootObject();
    auto tuples = (*unsafeCast<Vector<Handle<Map<KeyType, ValueType>>>>(hashTable))[workerID];

    // go through each key, value pair in the hash map we want to merge
    std::shared_ptr<std::list<std::pair<void*, void*>>> records = std::make_shared<std::list<std::pair<void*, void*>>>();
    for(auto it = tuples->begin(); it != tuples->end(); ++it) {
      records->emplace_back(&(*it).key, &(*it).value);
    }

    // return the tuples
    return std::move(records);
  }

private:

  /**
   * The id of the worker
   */
  size_t workerID = 0;

  /**
   * The number of records stored
   */
  atomic_uint64_t counts;

};

}

#endif //PDB_AGGREGATIONCOMBINERSINK_H
