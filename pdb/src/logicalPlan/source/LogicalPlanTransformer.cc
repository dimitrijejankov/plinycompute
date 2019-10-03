#include <SetScanner.h>
#include <AtomicComputationClasses.h>
#include <LogicalPlanTransformer.h>
#include <utility>
#include "LogicalPlan.h"

void pdb::Transformation::setPlan(pdb::LogicalPlanPtr planToSet) {
  logicalPlan = std::move(planToSet);
}

void pdb::Transformation::dropDependencies(const std::string &tupleSetName) {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // list of tuple sets to visit
  vector<std::string> tupleSetsToVisit = { tupleSetName };

  // stuff to remove
  std::vector<AtomicComputationPtr> toRemove;

  // while we have tuple sets
  while(!tupleSetsToVisit.empty()) {

    // get the input computation from the tuple sets to visit
    auto curComp = computations.getProducingAtomicComputation(tupleSetsToVisit.back());
    tupleSetsToVisit.pop_back();

    // the inputs of this computation
    std::vector<AtomicComputationPtr> inputs;

    // check if we have an input to this computation
    if(!curComp->getInput().isEmpty()) {

      // if we don't have two i
      inputs = { computations.getProducingAtomicComputation(curComp->getInput().getSetName()) };
    }

    // check if we have two inputs
    if(curComp->hasTwoInputs()) {
      inputs.emplace_back(computations.getProducingAtomicComputation(curComp->getRightInput().getSetName()));
    }

    // go through all the inputs
    for(const auto &input : inputs) {

      // we are free to kill this computation
      computations.removeConsumer(input->getOutput().getSetName(), curComp);

      // check if the input does not have any consumers if it does not we recurse further and schedule the comp for removal
      if(!computations.hasConsumer(input->getOutput().getSetName())) {

        // we need to visit the inputs next
        tupleSetsToVisit.emplace_back(input->getOutput().getSetName());

        // schedule the computation to be removed
        toRemove.emplace_back(input);
      }
    }
  }

  // remove all computations without consumers
  for(const auto &comp : toRemove) {
    computations.removeProducer(comp->getOutput().getSetName());
  }
}

void pdb::Transformation::dropDependents(const std::string &tupleSetName) {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // remove start computation from the consumers
  auto startComp = computations.getProducingAtomicComputation(tupleSetName);

  // remove the input
  computations.removeConsumer(startComp->getInputName(), startComp);

  // do we have two inputs here if so remove the right input
  if(startComp->hasTwoInputs()) {

    // remove the right side since there is one
    computations.removeConsumer(startComp->getRightInput().getSetName(), startComp);
  }

  // list of tuple sets to visit
  vector<std::string> tupleSetsToVisit = { tupleSetName };

  // the ones we need to remove
  vector<std::string> tupleSetsToRemove;

  // while we have tuple sets
  while(!tupleSetsToVisit.empty()) {

    // get the computation
    auto currComp = computations.getProducingAtomicComputation(tupleSetsToVisit.back());
    tupleSetsToVisit.pop_back();

    // insert the tuple set that we want to remove
    tupleSetsToRemove.emplace_back(currComp->getOutput().getSetName());

    // get all the consumers and visit them
    auto &consumers = computations.getConsumingAtomicComputations(currComp->getOutput().getSetName());
    for(const auto &c : consumers){

      // insert the consumer
      tupleSetsToVisit.emplace_back(c->getOutput().getSetName());
    }
  }

  // go through all of them
  for(const auto &ts : tupleSetsToRemove) {

    // remove the producer
    computations.removeProducer(ts);

    // remove all consumers
    computations.removeAllConsumers(ts);
  }
}


pdb::LogicalPlanTransformer::LogicalPlanTransformer(LogicalPlanPtr &logicalPlan) : logicalPlan(logicalPlan) {}

/**
 * Transformer Code
 */

void pdb::LogicalPlanTransformer::addTransformation(const pdb::TransformationPtr &transformation) {

  // set the logical plan
  transformation->setPlan(this->logicalPlan);

  // insert the transformation
  transformationsToApply.emplace_back(transformation);
}

pdb::LogicalPlanPtr pdb::LogicalPlanTransformer::applyTransformations() {

  // apply all the transformations
  for(auto &t : transformationsToApply) {
    if(t->canApply()) {
      t->apply();
    }
  }

  // return the transformed logical plan
  return logicalPlan;
}

/**
 * InsertKeyScanSetsTransformation Code
 */

pdb::InsertKeyScanSetsTransformation::InsertKeyScanSetsTransformation(std::string inputTupleSet) : inputTupleSet(std::move(inputTupleSet)) {}

void pdb::InsertKeyScanSetsTransformation::apply() {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // get the input computation
  auto inputComp = computations.getProducingAtomicComputation(inputTupleSet);

  // create a set scanner
  auto setScanner = std::make_shared<ScanSet>(inputComp->getOutput(),
                                              pageSetIdentifier.first,
                                              pageSetIdentifier.second,
                                              inputComp->getComputationName());

  // remove all the dependencies of this tuple set
  dropDependencies(inputTupleSet);

  // replace the atomic computation
  computations.replaceComputation(inputComp->getOutput().getSetName(), setScanner);

  // remove all the non used consumers
  computations.removeNonUsedConsumers();
}

bool pdb::InsertKeyScanSetsTransformation::canApply() {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // get the input computation
  auto inputComp = computations.getProducingAtomicComputation(inputTupleSet);

  // if it is not a scan set
  return inputComp->getAtomicComputationTypeID() != ScanSetAtomicTypeID;
}

/**
 * JoinFromKeyTransformation
 */

pdb::JoinKeySideTransformation::JoinKeySideTransformation(const std::string &inputTupleSet) : inputTupleSet(inputTupleSet) {

}

void pdb::JoinKeySideTransformation::apply() {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // get the input computation
  auto inputComp = computations.getProducingAtomicComputation(inputTupleSet);

  // get the key comp
  auto &keyComp = computations.getConsumingAtomicComputations(inputTupleSet).front();

  // get the input attribute
  auto inAttribute = inputComp->getOutput().getAtts().front();

  // get the key attribute
  auto keyAtts = TupleSpec::complement(keyComp->getOutput(), keyComp->getProjection());
  if(keyAtts.getAtts().size() != 1) {
    throw runtime_error("We were not able to figure out the key attribute");
  }

  // get the key attribute
  auto keyAttribute = keyAtts.getAtts().front();

  // insert the key and remove the input
  inputComp->getOutput().removeAtt(inAttribute);
  inputComp->getOutput().insertAtt(keyAttribute);

  // we modified the attributes sort them
  inputComp->sortOutput();

  // remove temporarily the scan set
  computations.removeProducer(inputComp->getOutput().getSetName());
  computations.replaceComputation(keyComp->getOutput().getSetName(), inputComp);

  // process all the way to the hash
  std::vector<AtomicComputationPtr> currentComps = computations.getConsumingAtomicComputations(inputComp->getOutput().getSetName());
  if(currentComps.size() != 1) {
    throw runtime_error("The join pipeline has branchings.");
  }

  // while we still have computations do stuff
  while (!currentComps.empty()) {

    // get the computation
    auto currComp = currentComps.back();
    currentComps.pop_back();

    // the input must not have the inAttribute otherwise we can not transform this
    if(currComp->getInput().hasAtt(inAttribute)) {
      throw runtime_error("The input attribute can not be applied unless it is a key extraction.");
    }

    // replace the input with the key
    currComp->getOutput().removeAtt(keyAttribute);
    currComp->getOutput().replaceAtt(inAttribute, keyAttribute);
    currComp->getProjection().removeAtt(keyAttribute);
    currComp->getProjection().replaceAtt(inAttribute, keyAttribute);

    // we modified the attributes sort them
    currComp->sortOutput();

    // check the consumers
    auto &consumers = computations.getConsumingAtomicComputations(currComp->getOutput().getSetName());
    if(consumers.size() != 1) {
      throw runtime_error("The join pipeline has branchings.");
    }

    // check if this is a join if it is stop
    if(consumers.front()->getAtomicComputationTypeID() == ApplyJoinTypeID) {
      break;
    }

    // insert the comps
    currentComps.emplace_back(consumers.front());
  }

  // remove all the non used consumers
  computations.removeNonUsedConsumers();
}

bool pdb::JoinKeySideTransformation::canApply() {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // get the input computation
  auto inputComp = computations.getProducingAtomicComputation(inputTupleSet);

  // if it is not a scan set we can not apply this
  if(inputComp->getAtomicComputationTypeID() != ScanSetAtomicTypeID) {
    return false;
  }

  // get the key comp
  auto consumers = computations.getConsumingAtomicComputations(inputTupleSet);
  if(consumers.size() != 1) {
    return false;
  }
  auto &keyComp = consumers.front();

  // check if this is an apply lambda
  if(keyComp->getAtomicComputationTypeID() == ApplyLambdaTypeID) {

    auto &kvPairs = (*keyComp->getKeyValuePairs());

    // get the lambda type
    auto it = kvPairs.find("lambdaType");
    return it != kvPairs.end() && it->second == "key";
  }

  // this is not an apply lambda with a key get out of there
  return false;
}


/**
 * JoinKeyTransformation
 */

pdb::JoinKeyTransformation::JoinKeyTransformation(const std::string &joinTupleSet) : joinTupleSet(joinTupleSet) {}

void pdb::JoinKeyTransformation::apply() {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // get the input computation
  auto joinComp = std::dynamic_pointer_cast<ApplyJoin>(computations.getProducingAtomicComputation(joinTupleSet));

  // get the output tuple set of the join
  auto lastOutput = joinComp->getOutput();

  // get the left and right input
  auto leftInput = computations.getProducingAtomicComputation(joinComp->getInput().getSetName());
  auto rightInput = computations.getProducingAtomicComputation(joinComp->getRightInput().getSetName());

  // get the key attributes
  auto leftKey = TupleSpec::complement(leftInput->getOutput(), joinComp->getInput());
  auto rightKey = TupleSpec::complement(rightInput->getOutput(), joinComp->getRightInput());

  // check if we have the key
  if(leftKey.getAtts().size() != 1 || rightKey.getAtts().size() != 1) {
    throw runtime_error("Could not find the key.");
  }

  // get the input attribute name of the join
  auto leftAtt = joinComp->getProjection();
  auto rightAtt = joinComp->getRightProjection();

  // check if we have the attribute
  if(leftAtt.getAtts().size() != 1 || rightAtt.getAtts().size() != 1) {
    throw runtime_error("Could not find the input attribute.");
  }

  // replace the left key
  joinComp->getProjection().replaceAtt(leftAtt.getAtts().front(), leftKey.getAtts().front());

  // replace the right key
  joinComp->getRightProjection().replaceAtt(rightAtt.getAtts().front(), rightKey.getAtts().front());

  // replace the stuff in the output
  joinComp->getOutput().replaceAtt(leftAtt.getAtts().front(), leftKey.getAtts().front());
  joinComp->getOutput().replaceAtt(rightAtt.getAtts().front(), rightKey.getAtts().front());

  // make this a key join
  joinComp->isKeyJoin = true;

  // we modified the attributes sort them
  joinComp->sortOutput();

  // get the name of the join computation
  std::string joinCompName = joinComp->getComputationName();

  // these are the attributes we need to remove and the aliases to the keys
  std::set<std::string> removeWhenApplied = { leftAtt.getAtts().front(), rightAtt.getAtts().front() };
  std::map<std::string, std::string> keyAliases;

  // process all the way to the hash
  std::vector<AtomicComputationPtr> currentComps = computations.getConsumingAtomicComputations(joinComp->getOutput().getSetName());
  if(currentComps.size() != 1) {
    throw runtime_error("The join pipeline has branchings.");
  }

  // basically the last computation that still remains after the transformation
  AtomicComputationPtr lastUsedComputation;

  // while we still have computations do stuff
  while (!currentComps.empty()) {

    // get the current computation
    auto currComp = currentComps.back();
    currentComps.pop_back();

    // save the current output tuple set since we are going to change it
    auto currOutput = currComp->getOutput();

    // check if any of the inputs to this computation comes from the input
    bool isDependentOnInput = std::any_of(currComp->getInput().getAtts().begin(),
                                          currComp->getInput().getAtts().end(),
                                          [&](auto &att) { return removeWhenApplied.find(att) != removeWhenApplied.end(); });

    // if it does depend in the input
    if(isDependentOnInput) {

      // figure out the generated columns
      auto generatedColumns = TupleSpec::complement(currOutput, lastOutput);

      // check if this computation is extracting the key
      if(currComp->getAtomicComputationTypeID() == ApplyLambdaTypeID && ((ApplyLambda*)currComp.get())->isExtractingKey()) {

        // figure out what key it is extracting
        if(currComp->getInput().hasAtt(leftAtt.getAtts().front())) {

          // mark the key alias
          keyAliases[generatedColumns.getAtts().front()] = leftKey.getAtts().front();

        } else if(currComp->getInput().hasAtt(rightAtt.getAtts().front())) {

          // mark the key alias
          keyAliases[generatedColumns.getAtts().front()] = rightKey.getAtts().front();
        }
        else {

          // so this is unexpected
          throw runtime_error("For some reason we are extracting the key from an attribute that is not the key.");
        }

      } else {

        // all the generated columns should be removed since they depend on the input and are not the key
        removeWhenApplied.insert(generatedColumns.getAtts().begin(), generatedColumns.getAtts().end());
      }

      // remove this one since we don't need it it depends on the input
      computations.removeAndRelink(currComp->getOutput().getSetName());
    }
    else {

      // remove input attributes from the output of the tuple set
      if(currComp->getOutput().hasAtt(leftAtt.getAtts().front())) {
        currComp->getOutput().removeAtt(leftAtt.getAtts().front());
      }

      if(currComp->getOutput().hasAtt(rightAtt.getAtts().front())) {
        currComp->getOutput().removeAtt(rightAtt.getAtts().front());
      }

      // remove input attributes from the projection
      if(currComp->getProjection().hasAtt(leftAtt.getAtts().front())) {
        currComp->getProjection().removeAtt(leftAtt.getAtts().front());
      }

      if(currComp->getProjection().hasAtt(rightAtt.getAtts().front())) {
        currComp->getProjection().removeAtt(rightAtt.getAtts().front());
      }

      // replace the key aliases
      for(const auto &keyAlias : keyAliases) {
        currComp->getOutput().replaceAtt(keyAlias.first, keyAlias.second);
        currComp->getInput().replaceAtt(keyAlias.first, keyAlias.second);
        currComp->getProjection().replaceAtt(keyAlias.first, keyAlias.second);
      }

      // if the keys are missing reinsert them into the output
      if(!currComp->getOutput().hasAtt(leftKey.getAtts().front())) {
        currComp->getOutput().insertAtt(leftKey.getAtts().front());
      }

      if(!currComp->getOutput().hasAtt(rightKey.getAtts().front())) {
        currComp->getOutput().insertAtt(rightKey.getAtts().front());
      }

      // if the keys are missing reinsert them into the projection
      if(!currComp->getProjection().hasAtt(leftKey.getAtts().front())) {
        currComp->getProjection().insertAtt(leftKey.getAtts().front());
      }

      if(!currComp->getProjection().hasAtt(rightKey.getAtts().front())) {
        currComp->getProjection().insertAtt(rightKey.getAtts().front());
      }

      // sort the output since we just modified the attributes
      currComp->sortOutput();

      // mark this as the last computation we actually used
      lastUsedComputation = currComp;
    }

    // get the consumers of this computation
    auto &cons = computations.getConsumingAtomicComputations(currComp->getOutput().getSetName());
    if(cons.size() != 1) {
      throw runtime_error("The join pipeline has branchings.");
    }

    if(cons.front()->getComputationName() == joinCompName) {

      // store the computation
      currentComps.emplace_back(cons.front());
    }

    // the last output is this output
    lastOutput = currOutput;
  }

  // remove the keys since we don't need them from the attributes
  if(lastUsedComputation != nullptr) {

    lastUsedComputation->getOutput().removeAtt(leftKey.getAtts().front());
    lastUsedComputation->getOutput().removeAtt(rightKey.getAtts().front());
    lastUsedComputation->getProjection().removeAtt(leftKey.getAtts().front());
    lastUsedComputation->getProjection().removeAtt(rightKey.getAtts().front());

    // sort the output since we just modified the attributes
    lastUsedComputation->sortOutput();
  }

  // remove all the non used consumers
  computations.removeNonUsedConsumers();
}

bool pdb::JoinKeyTransformation::canApply() {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // get the input computation
  auto joinComp = computations.getProducingAtomicComputation(joinTupleSet);

  // if it is not a join we can not apply this
  return joinComp->getAtomicComputationTypeID() == ApplyJoinTypeID;
}


/**
 * AggKeyTransformation
 */

pdb::AggKeyTransformation::AggKeyTransformation(const std::string &aggStartTupleSet) : aggStartTupleSet(aggStartTupleSet) {}

void pdb::AggKeyTransformation::apply() {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // get the start computation
  auto startComp = computations.getProducingAtomicComputation(aggStartTupleSet);

  // get the input
  auto startInput = computations.getProducingAtomicComputation(startComp->getInput().getSetName());

  // get the key input key from the join input
  auto generatedColumns = TupleSpec::complement(startComp->getOutput(), startComp->getProjection());
  if(generatedColumns.getAtts().size() != 1) {
    throw runtime_error("The join pipeline has branchings.");
  }
  auto keyAlias = generatedColumns.getAtts().front();

  // get the key input key from the join input
  if(startInput->getOutput().getAtts().size() != 1) {
    throw runtime_error("The join pipeline has branchings.");
  }
  auto inputKey = startInput->getOutput().getAtts().front();

  // get the key input key from the join input
  if(startComp->getInput().getAtts().size() != 1) {
    throw runtime_error("The join pipeline has branchings.");
  }
  auto recordColumn = startComp->getInput().getAtts().front();

  // init the start computation
  std::vector<AtomicComputationPtr> currentComps = computations.getConsumingAtomicComputations(startComp->getOutput().getSetName());
  if(currentComps.size() != 1) {
    throw runtime_error("The join pipeline has branchings.");
  }

  // remove and relink the key extraction
  computations.removeAndRelink(startComp->getOutput().getSetName());

  // try to find an apply aggregation
  std::set<std::string> toRemove = { recordColumn };
  while (!currentComps.empty()) {

    // get the current computation
    auto currComp = currentComps.back();
    currentComps.pop_back();

    // is this is an apply aggregation if it is we are fine
    if(currComp->getAtomicComputationTypeID() == ApplyAggTypeID) {

      // remove and relink the aggregation
      computations.removeAndRelink(currComp->getOutput().getSetName());
      break;
    }

    // does this computation depend on the record input
    auto dependsOnRecord = std::any_of(currComp->getInput().getAtts().begin(),
                                       currComp->getInput().getAtts().end(), [&toRemove]( const auto &c) {
      return toRemove.find(c) != toRemove.end();
    });

    generatedColumns = TupleSpec::complement(currComp->getOutput(), currComp->getProjection());

    // if this computation depends on the record input remove it
    if(dependsOnRecord) {

      // insert the generated columns
      toRemove.insert(generatedColumns.getAtts().begin(), generatedColumns.getAtts().end());

      // get the consumers of this computation
      auto &cons = computations.getConsumingAtomicComputations(currComp->getOutput().getSetName());
      currentComps.insert(currentComps.end(), cons.begin(), cons.end());

      // remove the computation
      computations.removeAndRelink(currComp->getOutput().getSetName());

      // we are done here we just removed this
      continue;
    }

    // remove all the unnecessary attributes
    for (auto &a : toRemove) {

      // just remove them
      currComp->getOutput().removeAtt(a);
      currComp->getProjection().removeAtt(a);
    }

    // replace the attributes
    currComp->getOutput().replaceAtt(keyAlias, inputKey);
    currComp->getProjection().replaceAtt(keyAlias, inputKey);
    currComp->getInput().replaceAtt(keyAlias, inputKey);

    // sort the output since we just modified the attributes
    currComp->sortOutput();

    // get the consumers of this computation
    auto &cons = computations.getConsumingAtomicComputations(currComp->getOutput().getSetName());
    currentComps.insert(currentComps.end(), cons.begin(), cons.end());
  }

  // remove all the non used consumers
  computations.removeNonUsedConsumers();
}

bool pdb::AggKeyTransformation::canApply() {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  // get the start computation
  auto startComp = computations.getProducingAtomicComputation(aggStartTupleSet);

  // make sure this is a key extraction lambda
  if(startComp->getAtomicComputationTypeID() != ApplyLambdaTypeID || !((ApplyLambda*)startComp.get())->isExtractingKey()) {
    return false;
  }

  // init the start computation
  std::vector<AtomicComputationPtr> currentComps = computations.getConsumingAtomicComputations(startComp->getOutput().getSetName());
  if(currentComps.size() != 1) {
    throw runtime_error("The join pipeline has branchings.");
  }

  // try to find an apply aggregation
  while (!currentComps.empty()) {

    // make sure we only have one computation
    if(currentComps.size() != 1) {
      return false;
    }

    // get the current computation
    auto currComp = currentComps.back();
    currentComps.pop_back();

    // is this is an apply aggregation if it is we are fine
    if(currComp->getAtomicComputationTypeID() == ApplyAggTypeID) {
      return true;
    }

    // get the consumers of this computation
    auto &cons = computations.getConsumingAtomicComputations(currComp->getOutput().getSetName());
    currentComps.insert(currentComps.end(), cons.begin(), cons.end());
  }

  return false;
}

/**
 * Drop Dependents
 */

pdb::DropDependents::DropDependents(const std::string &startTupleSet) : startTupleSet(startTupleSet) {}

void pdb::DropDependents::apply() {

  // drop all dependents of this tuple set
  dropDependents(startTupleSet);
}

bool pdb::DropDependents::canApply() {

  // get the computations from the plan
  auto &computations = logicalPlan->getComputations();

  return computations.getProducingAtomicComputation(startTupleSet) != nullptr;
}
