#include <physicalAlgorithms/TRAShuffleState.h>
#include <GenericWork.h>
#include <TRABlock.h>
#include "TRAShuffleStage.h"
#include "ExJob.h"

namespace pdb {

bool TRAShuffleStage::setup(const Handle<pdb::ExJob> &job,
                            const PDBPhysicalAlgorithmStatePtr &state,
                            const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                            const std::string &error) {
  // cast the state
  auto s = dynamic_pointer_cast<TRAShuffleState>(state);

  s->logger = make_shared<PDBLogger>("TRAShuffleStage_" + std::to_string(job->computationID));

  // the input page set
  s->index = storage->getIndex({0, inputPageSet});

  // grab the input page set
  s->inputPageSet = std::dynamic_pointer_cast<PDBRandomAccessPageSet>(storage->getPageSet({0, inputPageSet}));

  // get the receive page set
  s->feedingPageSet = storage->createFeedingAnonymousPageSet({0, "intermediate" },
                                                             1,
                                                             job->numberOfNodes);

  // create a random access page set
  s->outputPageSet = storage->createRandomAccessPageSet({0, sink});

  // the output index
  s->outputIndex = storage->createIndex({0, sink});

  /// 1. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes

  s->pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { s->pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  int32_t currNode = job->thisNode;
  s->senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->nodes[currNode]->port && job->nodes[i]->address == job->nodes[currNode]->address) {

      // make the self receiver
      s->selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(s->pageQueues->at(i), s->feedingPageSet, myMgr);
    }
    else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           1,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           s->logger,
                                                           std::make_pair(0, "intermediate" ),
                                                           s->pageQueues->at(i));

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      s->senders->emplace_back(sender);
    }
  }

  std::cout << "Setup\n";
  return true;
}

bool TRAShuffleStage::run(const Handle<pdb::ExJob> &job,
                          const PDBPhysicalAlgorithmStatePtr &state,
                          const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                          const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRAShuffleState>(state);

  // success indicator
  atomic_bool success;
  success = true;

  /// 0. Run the input scanner

  // create the buzzer
  atomic_int inputScanDone;
  inputScanDone = 0;
  PDBBuzzerPtr inputScanBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  auto bufferManager = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // repin all the pages
  s->inputPageSet->repinAll();

  // grab all the vectors
  std::vector<Handle<Vector<Handle<TRABlock>>>> inputVectors;
  for(int i = 0; i < s->inputPageSet->getNumPages(); ++i) {

    // get the vector from the page
    auto vec = ((Record<Vector<Handle<TRABlock>>> *) (*s->inputPageSet)[i]->getBytes())->getRootObject();
    inputVectors.push_back(vec);
  }


  // we use this as the hash pattern
  unordered_set<int32_t> pattern;
  for(int32_t i = 0; i < indices.size(); ++i) {
    pattern.insert(indices[i]);
  }

  // for each node we scan a part
  for(int32_t node = 0; node < job->nodes.size(); node++) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&inputScanDone, s, node, this, &inputVectors,
                                                                       job, bufferManager, &pattern](const PDBBuzzerPtr& callerBuzzer) {

      // find all the record from the index for this node
      std::vector<std::pair<int32_t, int32_t>> out;
      s->index->getWithHash(out, pattern, node, job->numberOfNodes);

      // get a new page
      auto currentPage = bufferManager->getPage();
      makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

      // is there stuff on the page
      bool stuffOnPage = false;

      // make the vector we write to
      Handle<Vector<Handle<pdb::TRABlock>>> writeMe = makeObject<Vector<Handle<pdb::TRABlock>>>();
      for(int i = 0; i < out.size();) {

        try {

          // grab the record index
          auto &recordIndex = out[i];

          // store it
          writeMe->push_back((*inputVectors[recordIndex.first])[recordIndex.second]);
          stuffOnPage = true;

          // go to the next record
          i++;

        } catch (pdb::NotEnoughSpace &n) {

          // make this the root object
          getRecord(writeMe);

          // insert into the page queue
          (*s->pageQueues)[node]->enqueue(currentPage);

          // grab a new page
          stuffOnPage = false;
          currentPage = bufferManager->getPage();
          makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

          // make a new vector!
          writeMe = makeObject<Vector<Handle<pdb::TRABlock>>>();
        }
      }

      // is there some stuff on the page
      if(stuffOnPage) {

        // make this the root object
        getRecord(writeMe);

        // insert into the page queue
        (*s->pageQueues)[node]->enqueue(currentPage);
      }

      // add a null so we notify that we are done
      (*s->pageQueues)[node]->enqueue(nullptr);

      // invalidate the block
      makeObjectAllocatorBlock(1024, true);

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, inputScanDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, inputScanBuzzer);
  }

  /// 1. Run the self receiver

  // create the buzzer
  atomic_int selfRecDone;
  selfRecDone = 0;
  PDBBuzzerPtr selfRefBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt = 1;
  });

  // run the work
  {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDone, s](const PDBBuzzerPtr& callerBuzzer) {

      // run the receiver
      if(s->selfReceiver->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, selfRecDone);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, selfRecDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, selfRefBuzzer);
  }

  /// 2. Run the senders

  // create the buzzer
  atomic_int sendersDone;
  sendersDone = 0;
  PDBBuzzerPtr sendersBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // go through each sender and run them
  for(auto &sender : *s->senders) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDone, sender, this](const PDBBuzzerPtr& callerBuzzer) {

      // run the sender
      if(sender->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendersDone);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, sendersDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, sendersBuzzer);
  }

  // I will kick off only one thread to make the index, if this happens to be an overhead we need to make it parallel.
  // create the buzzer
  atomic_int indexerDone;
  indexerDone = 0;
  PDBBuzzerPtr indexerBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt = 1;
  });

  {
    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&indexerDone, s](const PDBBuzzerPtr& callerBuzzer) {

      PDBPageHandle page;
      while((page = s->feedingPageSet->getNextPage(0)) != nullptr) {

        // store the page in the indexed page set
        auto loc = s->outputPageSet->pushPage(page);

        // get the vector from the page
        auto &vec = *(((Record<Vector<Handle<TRABlock>>> *) page->getBytes())->getRootObject());

        // generate the index
        for(int i = 0; i < vec.size(); ++i) {
          vec[i]->print();
          s->outputIndex->insert(*vec[i]->metaData, { loc,  i});
        }

        // unpin the page
        page->unpin();
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, indexerDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, indexerBuzzer);
  }

  while (inputScanDone != job->numberOfNodes) {
    inputScanBuzzer->wait();
  }

  while (selfRecDone != 1) {
    selfRefBuzzer->wait();
  }

  while (sendersDone != s->senders->size()) {
    sendersBuzzer->wait();
  }

  auto indexer_start = std::chrono::steady_clock::now();
  while (indexerDone != 1) {
    indexerBuzzer->wait();
  }
  auto indexer_end = std::chrono::steady_clock::now();

  // if this is too large we need to make indexing parallel
  std::cout << "Indexing overhead was " << std::chrono::duration_cast<std::chrono::nanoseconds>(indexer_end - indexer_start).count() << "[ns]" << '\n';

  std::cout << "run\n";
  return true;
}

void TRAShuffleStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                              const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  // cast the state
  auto s = dynamic_pointer_cast<TRAShuffleState>(state);

  storage->removePageSet({0, "intermediate" });

  std::cout << "cleanup\n";
}

TRAShuffleStage::TRAShuffleStage(const std::string &inputPageSet,
                                 const std::string &sink,
                                 const pdb::Vector<int32_t> &indices) :
     PDBPhysicalAlgorithmStage(*(_sink),
                               *(_sources),
                               *(_finalTupleSet),
                               *(_secondarySources),
                               *(_setsToMaterialize)), inputPageSet(inputPageSet), sink(sink), indices(indices) {}


}

const pdb::PDBSinkPageSetSpec *pdb::TRAShuffleStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRAShuffleStage::_sources = nullptr;
const pdb::String *pdb::TRAShuffleStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRAShuffleStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRAShuffleStage::_setsToMaterialize = nullptr;