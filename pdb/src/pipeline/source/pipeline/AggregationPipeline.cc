#include <utility>

#include <utility>

//
// Created by dimitrije on 3/27/19.
//

#include <AggregationPipeline.h>
#include <MemoryHolder.h>

void pdb::AggregationPipeline::run() {

  // this is where we are outputting all of our results to
  MemoryHolderPtr myRAM = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());

  // create an output container create it.
  myRAM->outputSink = merger->createNewOutputContainer();

  // aggregate all hash maps
  PDBPageHandle inputPage;
  while ((inputPage = inputPageSet->getNextPage(workerID)) != nullptr) {

    // write out the page
    merger->writeOutPage(inputPage, myRAM->outputSink);
  }

  // increment the records in the output set
  if(myRAM->outputSink != nullptr) {
    outputPageSet->increaseRecords(merger->getNumRecords(myRAM->outputSink));
  }

  // we only have one iteration
  myRAM->setIteration(0);

  // and force the reference count for this guy to go to zero
  myRAM->outputSink.emptyOutContainingBlock();

  // unpin the page so we don't have problems
  myRAM->pageHandle->unpin();

  // TODO make this nicer
  makeObjectAllocatorBlock(1024, true);
}

pdb::AggregationPipeline::AggregationPipeline(size_t workerID,
                                              pdb::PDBAnonymousPageSetPtr outputPageSet,
                                              pdb::PDBAbstractPageSetPtr inputPageSet,
                                              pdb::ComputeSinkPtr merger) : workerID(workerID),
                                                                            outputPageSet(std::move(outputPageSet)),
                                                                            inputPageSet(std::move(inputPageSet)),
                                                                            merger(std::move(merger)) {}
