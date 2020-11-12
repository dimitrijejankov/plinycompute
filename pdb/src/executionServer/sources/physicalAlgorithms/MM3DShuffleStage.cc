#include <physicalAlgorithms/MM3DState.h>
#include <GenericWork.h>
#include <TRABlock.h>
#include "MM3DShuffleStage.h"
#include "MM3DState.h"
#include "PDBSetObject.h"
#include "ExJob.h"

pdb::MM3DShuffleStage::MM3DShuffleStage(int32_t n, int32_t num_nodes, int32_t num_threads) :

    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), idx{.num_nodes = num_nodes,
                                                          .num_threads = num_threads,
                                                          .n = n} {
}

bool pdb::MM3DShuffleStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                  const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                  const shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                  const string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<pdb::MM3DState>(state);

  //
  to_send_lhs.resize(idx.num_nodes);
  to_send_rhs.resize(idx.num_nodes);

  //
  s->lhsSet = std::dynamic_pointer_cast<pdb::PDBRandomAccessPageSet>(storage->getPageSet({0, "myData:A"}));

  //
  s->rhsSet = std::dynamic_pointer_cast<pdb::PDBRandomAccessPageSet>(storage->getPageSet({0, "myData:B"}));

  //
  s->pageQueuesLHS = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { s->pageQueuesLHS->emplace_back(std::make_shared<PDBPageQueue>()); }

  s->pageQueuesRHS = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { s->pageQueuesRHS->emplace_back(std::make_shared<PDBPageQueue>()); }

  ///

  // get the receive page set
  s->feedingPageSetLHS = storage->createFeedingAnonymousPageSet({0, "intermediateLHS" },
                                                             1,
                                                             job->numberOfNodes);

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  int32_t currNode = job->thisNode;
  s->sendersLHS = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->nodes[currNode]->port && job->nodes[i]->address == job->nodes[currNode]->address) {

      // make the self receiver
      s->selfReceiverLHS = std::make_shared<pdb::PDBPageSelfReceiver>(s->pageQueuesLHS->at(i), s->feedingPageSetLHS, myMgr);
    }
    else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           1,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           s->logger,
                                                           std::make_pair(0, "intermediateLHS" ),
                                                           s->pageQueuesLHS->at(i));

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      s->sendersLHS->emplace_back(sender);
    }
  }

  ///

  // get the receive page set
  s->feedingPageSetRHS = storage->createFeedingAnonymousPageSet({0, "intermediateRHS" },
                                                             1,
                                                             job->numberOfNodes);

  s->sendersRHS = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->nodes[currNode]->port && job->nodes[i]->address == job->nodes[currNode]->address) {

      // make the self receiver
      s->selfReceiverRHS = std::make_shared<pdb::PDBPageSelfReceiver>(s->pageQueuesRHS->at(i), s->feedingPageSetRHS, myMgr);
    }
    else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           1,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           s->logger,
                                                           std::make_pair(0, "intermediateRHS" ),
                                                           s->pageQueuesRHS->at(i));

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      s->sendersRHS->emplace_back(sender);
    }
  }

  return true;
}

bool pdb::MM3DShuffleStage::run(const pdb::Handle<pdb::ExJob> &job,
                                const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                const shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                const string &error) {
  // cast the state
  auto s = dynamic_pointer_cast<pdb::MM3DState>(state);

  // success indicator
  atomic_bool success;

  // repin the sets
  s->lhsSet->repinAll();
  s->rhsSet->repinAll();

  // create the buzzer
  atomic_int sendingDone;
  PDBBuzzerPtr sendingBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  std::cout << "A : " << '\n';

  // go through all the records that we need to send
  for(int32_t pageNum = 0; pageNum < s->lhsSet->getNumPages(); ++pageNum) {

    // get the actual page
    auto page = (*s->lhsSet)[pageNum];

    // get the vector from the page
    auto &vec = *(((Record<Vector<Handle<TRABlock>>> *) page->getBytes())->getRootObject());

    // go through the vector
    for(int i = 0; i < vec.size(); ++i) {

      auto x = vec[i]->getkey0();
      auto z = vec[i]->getkey1();

      // go through every y we need to send this
      for(int y = 0; y < idx.get_side(); ++y) {

        // get the node id
        auto node_id = idx.get(x, y, z);

        // check if there we already are sending this to a node
        if(to_send_lhs[std::get<0>(node_id)].find(&vec[i]) == to_send_lhs[std::get<0>(node_id)].end()) {

          std::cout << "row ID : " << x << " col ID : " << z << " (" << std::get<0>(node_id) << " - " << x << ", " << y << ", " << z << ')' << '\n';

          // where do we need to send
          to_send_lhs[std::get<0>(node_id)].insert(&vec[i]);
        }
      }
    }
  }

  std::cout << "B : " << '\n';
  // go through all the records that we need to send
  for(int32_t pageNum = 0; pageNum < s->rhsSet->getNumPages(); ++pageNum) {

    // get the actual page
    auto page = (*s->rhsSet)[pageNum];

    // get the vector from the page
    auto &vec = *(((Record<Vector<Handle<TRABlock>>> *) page->getBytes())->getRootObject());

    // go through the vector
    for(int i = 0; i < vec.size(); ++i) {

      //
      auto z = vec[i]->getkey0();
      auto y = vec[i]->getkey1();

      // go through every y we need to send this

      for(int x = 0; x < idx.get_side(); ++x) {

        // get the node id
        auto node_id = idx.get(x, y, z);

        if(to_send_rhs[std::get<0>(node_id)].find(&vec[i]) == to_send_rhs[std::get<0>(node_id)].end()) {

          std::cout << "row ID : " << z << " col ID : " << y << " (" << std::get<0>(node_id) << " - " << x << ", " << y << ", " << z << ')' << '\n';
          // where do we need to send
          to_send_rhs[std::get<0>(node_id)].insert(&vec[i]);
        }
      }
    }
  }

  // create the buzzer
  atomic_int prepDone;
  prepDone = 0;
  PDBBuzzerPtr prepBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // prepare the pages we are going to send
  auto bufferManager = storage->getFunctionalityPtr<PDBBufferManagerInterface>();
  for(int node = 0; node < job->numberOfNodes; ++node) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([node, &bufferManager, &prepDone, this, s](const PDBBuzzerPtr& callerBuzzer) {

      // get a new page
      auto currentPage = bufferManager->getPage();
      makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

      // make the vector we write to
      Handle<Vector<Handle<pdb::TRABlock>>> writeMe = makeObject<Vector<Handle<pdb::TRABlock>>>();
      auto it = to_send_lhs[node].begin();
      while(it != to_send_lhs[node].end()) {

        try {

          // store it
          writeMe->push_back(**it);

          // go to the next record
          it++;

        } catch (pdb::NotEnoughSpace &n) {

          // make this the root object
          getRecord(writeMe);

          // insert into the page queue
          (*s->pageQueuesLHS)[node]->enqueue(currentPage);

          // grab a new page
          currentPage = bufferManager->getPage();
          makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

          // make a new vector!
          writeMe = makeObject<Vector<Handle<pdb::TRABlock>>>();
        }
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, prepDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, prepBuzzer);
  }

  // the senders for rhs
  for(int node = 0; node < job->numberOfNodes; ++node) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([node, &bufferManager, &prepDone, this, s](const PDBBuzzerPtr& callerBuzzer) {

      // get a new page
      auto currentPage = bufferManager->getPage();
      makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

      // make the vector we write to
      Handle<Vector<Handle<pdb::TRABlock>>> writeMe = makeObject<Vector<Handle<pdb::TRABlock>>>();
      auto it = to_send_rhs[node].begin();
      while(it != to_send_rhs[node].end()) {

        try {

          // store it
          writeMe->push_back(**it);

          // go to the next record
          it++;

        } catch (pdb::NotEnoughSpace &n) {

          // make this the root object
          getRecord(writeMe);

          // insert into the page queue
          (*s->pageQueuesRHS)[node]->enqueue(currentPage);

          // grab a new page
          currentPage = bufferManager->getPage();
          makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

          // make a new vector!
          writeMe = makeObject<Vector<Handle<pdb::TRABlock>>>();
        }
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, prepDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, prepBuzzer);
  }

  ////

  /// Run the self receiver LHS

  // create the buzzer
  atomic_int selfRecDoneLHS;
  selfRecDoneLHS = 0;
  PDBBuzzerPtr selfRefBuzzerLHS = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDoneLHS, s](const PDBBuzzerPtr& callerBuzzer) {

      // run the receiver
      if(s->selfReceiverLHS->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, selfRecDoneLHS);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, selfRecDoneLHS);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, selfRefBuzzerLHS);
  }

  /// 2. Run the senders LHS

  // create the buzzer
  atomic_int sendersDoneLHS;
  sendersDoneLHS = 0;
  PDBBuzzerPtr sendersBuzzerLHS = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // go through each sender and run them
  for(auto &sender : *s->sendersLHS) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDoneLHS, sender, this](const PDBBuzzerPtr& callerBuzzer) {

      // run the sender
      if(sender->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendersDoneLHS);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, sendersDoneLHS);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, sendersBuzzerLHS);
  }

  ////

  /// Run the self receiver,

  // create the buzzer
  atomic_int selfRecDoneRHS;
  selfRecDoneRHS = 0;
  PDBBuzzerPtr selfRefBuzzerRHS = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDoneRHS, s](const PDBBuzzerPtr& callerBuzzer) {

      // run the receiver
      if(s->selfReceiverRHS->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, selfRecDoneRHS);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, selfRecDoneRHS);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, selfRefBuzzerRHS);
  }

  /// 2. Run the senders

  // create the buzzer
  atomic_int sendersDoneRHS;
  sendersDoneRHS = 0;
  PDBBuzzerPtr sendersBuzzerRHS = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // go through each sender and run them
  for(auto &sender : *s->sendersRHS) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDoneRHS, sender, this](const PDBBuzzerPtr& callerBuzzer) {

      // run the sender
      if(sender->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendersDoneRHS);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, sendersDoneRHS);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, sendersBuzzerRHS);
  }

  // check if the preparation is done
  while (prepDone != 2 * job->numberOfNodes) {
    prepBuzzer->wait();
  }

  // add nulls to it
  for(int i = 0; i < s->pageQueuesLHS->size(); ++i) {
    (*s->pageQueuesLHS)[i]->enqueue(nullptr);
  }

  // add null to it
  for(int i = 0; i < s->pageQueuesRHS->size(); ++i) {
    (*s->pageQueuesRHS)[i]->enqueue(nullptr);
  }

  // wait while we are running the receiver
  while(selfRecDoneLHS == 1) {
    selfRefBuzzerLHS->wait();
  }
  while(selfRecDoneRHS == 1) {
    selfRefBuzzerRHS->wait();
  }

  // wait while we are running the senders
  while(sendersDoneLHS < s->sendersLHS->size()) {
    sendersBuzzerLHS->wait();
  }

  // wait while we are running the senders
  while(sendersDoneRHS < s->sendersRHS->size()) {
    sendersBuzzerRHS->wait();
  }

  return true;
}

void pdb::MM3DShuffleStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                    const shared_ptr<pdb::PDBStorageManagerBackend> &storage) {}

const pdb::PDBSinkPageSetSpec *pdb::MM3DShuffleStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::MM3DShuffleStage::_sources = nullptr;
const pdb::String *pdb::MM3DShuffleStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::MM3DShuffleStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::MM3DShuffleStage::_setsToMaterialize = nullptr;