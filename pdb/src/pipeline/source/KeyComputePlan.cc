#include <utility>

#include <KeyComputePlan.h>
#include <processors/NullProcessor.h>

pdb::KeyComputePlan::KeyComputePlan(pdb::LogicalPlanPtr myPlan) : ComputePlan(std::move(myPlan)) {}

std::vector<AtomicComputationPtr> pdb::KeyComputePlan::getLeftPipelineComputations(AtomicComputationPtr &source) {

  // we are going to return this
  std::vector<AtomicComputationPtr> tmp;

  // get the current tuple set
  auto currTupleSet = source->getOutputName();

  // while we don't hit the join computation loop
  while(!myPlan->getComputations().getConsumingAtomicComputations(currTupleSet).empty()) {

    // get all the consumers
    auto &consumers = myPlan->getComputations().getConsumingAtomicComputations(currTupleSet);

    // there has to be exactly one consumer otherwise something went wrong
    if(consumers.size() != 1 ) {
      return {};
    }

    // insert the consumer
    tmp.emplace_back(consumers.back());

    // check if this is a join we are done
    auto compType = consumers.back()->getAtomicComputationTypeID();
    if(compType == HashOneTypeID || compType == HashLeftTypeID || compType == HashRightTypeID) {
      return std::move(tmp);
    }

    // update the tuple set for the next iteration
    currTupleSet = consumers.back()->getOutputName();
  }

  // we did not find a hash, return empty list
  return {};
}

pdb::PipelinePtr pdb::KeyComputePlan::buildHashPipeline(AtomicComputationPtr &source,
                                                        const pdb::PDBAbstractPageSetPtr &inputPageSet,
                                                        const pdb::PDBAnonymousPageSetPtr &outputPageSet,
                                                        map<pdb::ComputeInfoType, pdb::ComputeInfoPtr> &params) {

  // get the left pipeline computations
  auto listSoFar = getLeftPipelineComputations(source);

  // if there is nothing in the pipeline finish
  if(listSoFar.empty()) {
    return nullptr;
  }

  /// 1. Figure out the source

  // our source is a normal source and not a join source, so we just grab it from the computation
  auto computeSource = myPlan->getNode(source->getComputationName()).getComputation().getComputeSource(inputPageSet, std::numeric_limits<size_t>::max(), 0, params);

  /// 2. Figure out the sink

  // get the atomic computation of the hasher
  auto &targetAtomicComp = listSoFar.back();

  // get the computation that corresponds to the hasher
  auto targetComputationName = targetAtomicComp->getComputationName();

  // returns the input specifier
  auto specifier = getSinkSpecifier(targetAtomicComp, targetComputationName);

  // returns the key sink
  auto computeSink = myPlan->getNode(targetComputationName).getComputation().getKeySink(std::get<0>(specifier),
                                                                                        std::get<1>(specifier),
                                                                                        std::get<2>(specifier),
                                                                                        1,
                                                                                        params,
                                                                                        myPlan);

  /// 3. Assemble the pipeline

  // assemble the whole pipeline
  return assemblePipeline(source->getOutputName(),
                          outputPageSet,
                          computeSource,
                          computeSink,
                          std::make_shared<NullProcessor>(),
                          params,
                          listSoFar,
                          1,
                          1,
                          0);
}

