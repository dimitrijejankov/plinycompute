//
// Created by dimitrije on 5/7/19.
//

#include <ComputePlan.h>

#include <physicalAlgorithms/PDBShuffleForJoinAlgorithm.h>
#include <ExJob.h>
#include <PDBStorageManagerBackend.h>
#include <PDBPageNetworkSender.h>
#include <ShuffleJoinProcessor.h>
#include <PDBPageSelfReceiver.h>
#include <GenericWork.h>
#include <memory>

pdb::PDBShuffleForJoinAlgorithm::PDBShuffleForJoinAlgorithm(const std::string &firstTupleSet,
                                                            const std::string &finalTupleSet,
                                                            const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                            const pdb::Handle<PDBSinkPageSetSpec> &intermediate,
                                                            const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                            const pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &secondarySources,
                                                            const bool swapLHSandRHS)
                                                            : PDBPhysicalAlgorithm(firstTupleSet, finalTupleSet, source, sink, secondarySources, swapLHSandRHS),
                                                              intermediate(intermediate) {

}

pdb::PDBPhysicalAlgorithmType pdb::PDBShuffleForJoinAlgorithm::getAlgorithmType() {
  return ShuffleForJoin;
}

bool pdb::PDBShuffleForJoinAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // success indicator
  atomic_bool success;
  success = true;

  /// 1. Run the self receiver,

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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDone, this](PDBBuzzerPtr callerBuzzer) {

      // run the receiver
      if(selfReceiver->run()) {

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
  for(auto &sender : *senders) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDone, sender, this](PDBBuzzerPtr callerBuzzer) {

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

  /// 3. Run the join pipelines

  // create the buzzer
  atomic_int joinCounter;
  joinCounter = 0;
  PDBBuzzerPtr joinBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // here we get a worker per pipeline and run all the shuffle join side pipelines.
  for (int workerID = 0; workerID < joinShufflePipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&joinCounter, workerID, this](PDBBuzzerPtr callerBuzzer) {

      // run the pipeline
      (*joinShufflePipelines)[workerID]->run();

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, joinCounter);
    });

    // run the work
    worker->execute(myWork, joinBuzzer);
  }

  // wait until all the shuffle join side pipelines have completed
  while (joinCounter < joinShufflePipelines->size()) {
    joinBuzzer->wait();
  }

  // ok they have finished now push a null page to each of the preagg queues
  for(auto &queue : *pageQueues) { queue->enqueue(nullptr); }

  // wait while we are running the receiver
  while(selfRecDone == 0) {
    selfRefBuzzer->wait();
  }

  // wait while we are running the senders
  while(sendersDone < senders->size()) {
    sendersBuzzer->wait();
  }

  return true;
}

bool pdb::PDBShuffleForJoinAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) {

  // init the plan
  ComputePlan plan(job->tcap, *job->computations);
  logicalPlan = plan.getPlan();

  // init the logger
  logger = make_shared<PDBLogger>("joinPipeAlgorithm" + std::to_string(job->computationID));

  /// 1. Figure out the source page set

  // get the source computation
  auto srcNode = logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if(srcNode->getAtomicComputationTypeID() == ScanSetAtomicTypeID) {

    // cast it to a scan
    auto scanNode = std::dynamic_pointer_cast<ScanSet>(srcNode);

    // get the page set
    sourcePageSet = storage->createPageSetFromPDBSet(scanNode->getDBName(),
                                                     scanNode->getSetName(),
                                                     std::make_pair(source->pageSetIdentifier.first, source->pageSetIdentifier.second));
  }
  else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->getPageSet(std::make_pair(source->pageSetIdentifier.first, source->pageSetIdentifier.second));
    sourcePageSet->resetPageSet();
  }

  // did we manage to get a source page set? if not the setup failed
  if(sourcePageSet == nullptr) {
    return false;
  }

  /// 2. Make the intermediate page set

  // get the sink page set
  auto intermediatePageSet = storage->createAnonymousPageSet(intermediate->pageSetIdentifier);

  // did we manage to get a sink page set? if not the setup failed
  if (intermediatePageSet == nullptr) {
    return false;
  }

  /// 3. Init the shuffle queues

  pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

  // set the parameters
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();
  std::map<ComputeInfoType, ComputeInfoPtr> params = { { ComputeInfoType::PAGE_PROCESSOR, plan.getProcessorForJoin(finalTupleSet,
                                                                                                                   job->numberOfNodes,
                                                                                                                   job->numberOfProcessingThreads,
                                                                                                                   *pageQueues,
                                                                                                                   myMgr) } };

  /// 4. Create the page set that contains the shuffled join side pages for this node

  // get the receive page set
  auto recvPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second),
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

  // did we manage to get a page set where we receive this? if not the setup failed
  if(recvPageSet == nullptr) {
    return false;
  }

  /// 5. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes

  senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->thisNode->port && job->nodes[i]->address == job->thisNode->address) {

      // make the self receiver
      selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(pageQueues->at(i), recvPageSet);
    }
    else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           logger,
                                                           std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second),
                                                           pageQueues->at(i));

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      senders->emplace_back(sender);
    }
  }

  /// 6. Create the join pipeline

  joinShufflePipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t workerID = 0; workerID < job->numberOfProcessingThreads; ++workerID) {

    // build the join pipeline
    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       intermediatePageSet,
                                       params,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       workerID);

    // store the join pipeline
    joinShufflePipelines->push_back(pipeline);
  }

  return true;
}

void pdb::PDBShuffleForJoinAlgorithm::cleanup() {

  // invalidate everything
  pageQueues = nullptr;
  joinShufflePipelines = nullptr;
  logger = nullptr;
  selfReceiver = nullptr;
  senders = nullptr;
  intermediate = nullptr;
  logicalPlan = nullptr;
}
