#include <utility>

#include <utility>

//
// Created by dimitrije on 3/27/19.
//

#include <AggregationPipeline.h>
#include <MemoryHolder.h>
#include <GenericWork.h>

void pdb::AggregationPipeline::run() {

  // the forewarding queue to the child
  AggregationCombinerSinkBase::record_forwarding_queue_t child_queue;

  // the buzzer we use for the child
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {});

  // get a worker from the server
  pdb::PDBWorkerPtr worker = workerQueue->getWorker();

  // start a child thread to do the processing
  pdb::PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&](const PDBBuzzerPtr& callerBuzzer) {
    merger->processing_thread(outputPageSet, child_queue, workerQueue);
  });

  // run the work
  worker->execute(myWork, tempBuzzer);

  PDBPageHandle inputPage;
  while ((inputPage = inputPageSet->getNextPage(workerID)) != nullptr) {

    // repin the page
    inputPage->repin();

    // get the records
    auto records = merger->getTupleList(inputPage);

    // process
    child_queue.enqueue(records);

    // wait till everything is processed
    child_queue.wait_till_processed();

    // unpin the page
    inputPage->unpin();
  }

  // mark that we are done
  child_queue.enqueue(nullptr);

  // wait till we have processed everything
  child_queue.wait_till_processed();
}

pdb::AggregationPipeline::AggregationPipeline(size_t workerID,
                                              PDBAnonymousPageSetPtr outputPageSet,
                                              PDBAbstractPageSetPtr inputPageSet,
                                              PDBWorkerQueuePtr workerQueue,
                                              const pdb::ComputeSinkPtr &merger) : workerID(workerID),
                                                                                   outputPageSet(std::move(outputPageSet)),
                                                                                   workerQueue(std::move(workerQueue)),
                                                                                   inputPageSet(std::move(inputPageSet)) {
  this->merger = std::dynamic_pointer_cast<AggregationCombinerSinkBase>(merger);
}
