#include <physicalAlgorithms/TRAShuffleState.h>
#include "TRAShuffleStage.h"
#include "ExJob.h"

namespace pdb {

bool TRAShuffleStage::setup(const Handle<pdb::ExJob> &job,
                            const PDBPhysicalAlgorithmStatePtr &state,
                            const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                            const std::string &error) {
  // cast the state
  auto s = dynamic_pointer_cast<TRAShuffleState>(state);

  // the input page set
  s->index = storage->getIndex({0, inputPageSet});

  // grab the input page set
  s->inputPageSet = storage->getPageSet({0, inputPageSet});

  // get the receive page set
  s->feedingPageSet = storage->createFeedingAnonymousPageSet({0, "intermediate" },
                                                             1,
                                                             job->numberOfNodes);

  // create a random access page set
  s->outputPageSet = storage->createRandomAccessPageSet({0, sink});

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

  std::cout << "run\n";
  return true;
}

void TRAShuffleStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                              const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  // cast the state
  auto s = dynamic_pointer_cast<TRAShuffleState>(state);

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