#include <utility>

#include <utility>

//
// Created by dimitrije on 4/11/19.
//

#include <JoinBroadcastPipeline.h>
#include <pipeline/JoinBroadcastPipeline.h>
#include <MemoryHolder.h>

pdb::JoinBroadcastPipeline::JoinBroadcastPipeline(size_t workerID,
                                                  pdb::PDBAnonymousPageSetPtr outputPageSet,
                                                  pdb::PDBAbstractPageSetPtr inputPageSet,
                                                  pdb::ComputeSinkPtr merger)
    : workerID(workerID),
      outputPageSet(std::move(outputPageSet)),
      inputPageSet(std::move(inputPageSet)),
      merger(std::move(merger)) {}

void pdb::JoinBroadcastPipeline::run() {

  // this is where we are outputting all of our results to
  MemoryHolderPtr myRAM = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());

  // aggregate all hash maps
  PDBPageHandle inputPage;
  while ((inputPage = inputPageSet->getNextPage(workerID)) != nullptr) {

    // if we haven't created an output container create it.
    if (myRAM->outputSink == nullptr) {
      myRAM->outputSink = merger->createNewOutputContainer();
      inputPage->unpin();
    }
    // write out the page
    merger->writeOutPage(inputPage, myRAM->outputSink);

    inputPage->unpin();
  }

  // make sure we have a root record on the page
  getRecord(myRAM->outputSink);

  // we only have one iteration
  myRAM->setIteration(0);

  // and force the reference count for this guy to go to zero
  myRAM->outputSink.emptyOutContainingBlock();

  // unpin the page so we don't have problems
  myRAM->pageHandle->unpin();

  // TODO make this nicer
  makeObjectAllocatorBlock(1024, true);
}
