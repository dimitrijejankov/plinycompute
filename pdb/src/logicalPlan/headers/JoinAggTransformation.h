#include <utility>

#include <utility>

#pragma once

#include <LogicalPlan.h>
#include "AtomicComputationClasses.h"

namespace pdb {

class JoinAggTransformation {
public:

  JoinAggTransformation(LogicalPlanPtr logicalPlan,
                        std::string leftSide,
                        std::pair<std::string, std::string> leftSet,
                        std::string rightSide,
                        std::pair<std::string, std::string> rightSet,
                        std::string join,
                        std::string agg) : logicalPlan(std::move(logicalPlan)),
                                           leftSide(std::move(leftSide)),
                                           leftSet(std::move(leftSet)),
                                           rightSide(std::move(rightSide)),
                                           rightSet(std::move(rightSet)),
                                           join(std::move(join)),
                                           agg(std::move(agg)) {};

  void transform() {

    /**
     * 1. Figure out the LHS
     */

    // the computations
    auto computations = logicalPlan->computations;

    // try to find the left scan set
    auto it = std::find_if(computations.getAllScanSets().begin(), computations.getAllScanSets().end(), [&](auto &set) {
      std::cout << set->getOutputName() << " : " << leftSide << "\n";
      return set->getOutputName() == leftSide;
    });
    if(it == computations.getAllScanSets().end()) {
      throw std::runtime_error("Failed to find the left scan set");
    }

    // make the left side
    leftAtomicComputations = { *it };

    // get the next computation
    auto &tmp = computations.getConsumingAtomicComputations(leftSide);

    // try to find to the left side
    while (true) {

      //
      if(tmp.size() != 1) {
        throw std::runtime_error("For some reason the left side is split");
      }
      auto &comp = tmp.front();

      // make sure the
      if(comp->getOutputName() == join) {
        break;
      }

      // add the computation to the left side
      leftAtomicComputations.emplace_back(comp);

      // get the next consuming atomic computation
      tmp = computations.getConsumingAtomicComputations(comp->getOutputName());
    }

    // try to find the right scan set
    it = std::find_if(computations.getAllScanSets().begin(), computations.getAllScanSets().end(), [&](auto &set) {
      return set->getOutputName() == rightSide;
    });
    if(it == computations.getAllScanSets().end()) {
      throw std::runtime_error("Failed to find the right scan set");
    }

    /**
     * 2. Figure out the RHS
     */

    // make the right side
    rightAtomicComputations = { *it };

    tmp = computations.getConsumingAtomicComputations(rightSide);
    while (true) {

      //
      if(tmp.size() != 1) {
        throw std::runtime_error("For some reason the right side is split");
      }
      auto &comp = tmp.front();

      // make sure the
      if(comp->getOutputName() == agg) {
        break;
      }

      // add the computation to the left side
      rightAtomicComputations.emplace_back(comp);

      // get the next consuming atomic computation
      tmp = computations.getConsumingAtomicComputations(comp->getOutputName());
    }

    /**
     * 3. Transform the LHS
     */

    transformLHS();

    /**
     * 4. Transform the RHS
     */
    transformRHS();

    std::cout << "Left\n";
    for(auto &c : leftAtomicComputations) {
      std::cout << *c;
    }

    std::cout << "Right\n";
    for(auto &c : rightAtomicComputations) {
      std::cout << *c;
    }
  }

protected:

  void transformRHS() {

    /**
     * 1. Extract the input name and the key
     */

    std::vector<AtomicComputationPtr> outputRHS;

    // get the scan set
    auto curr = rightAtomicComputations.begin();

    // get the input attribute
    rightIn = (*curr)->getOutput().getAtts().front();

    // go the the key extraction and make sure it is a key extraction
    curr++;
    auto &kvPairs = (*curr)->getKeyValuePairs();
    auto kvIt = kvPairs->find("lambdaType");
    if(kvIt == kvPairs->end() || kvIt->second != "key") {
      throw runtime_error("There is not key extraction lambda after the scan set.");
    }

    // get the key attribute
    auto gen = (*curr)->getGeneratedColumns();
    if(gen.size() != 1) {
      throw runtime_error("The key extraction should only generate one key!");
    }
    rightKey = gen.front();

    // add the scan set for key
    outputRHS.emplace_back(std::make_shared<ScanSet>(TupleSpec{(*curr)->getOutputName(), { rightKey }},
                                                     rightSet.first,
                                                     rightSet.second,
                                                     (*curr)->getComputationName()));

    /**
     * 2. Modify the left side, basically the rhs input and keep only the rhs key
     */

    curr++;
    for(; curr != rightAtomicComputations.end(); curr++) {

      auto &comp = *curr;

      if(comp->getAtomicComputationTypeID() == ApplyJoinTypeID) {
        break;
      }

      auto &outAtts = comp->getOutput().getAtts();
      auto &projectionAtts = comp->getProjection().getAtts();
      auto &inputAtts = comp->getInput().getAtts();

      // remove the input
      auto it = std::find(outAtts.begin(), outAtts.end(), rightIn);
      outAtts.erase(it);

      // remove the from the input attributes
      it = std::find(projectionAtts.begin(), projectionAtts.end(), rightIn);
      projectionAtts.erase(it);

      // the key was removed add it
      it = std::find(outAtts.begin(), outAtts.end(), rightKey);
      if(it == outAtts.end()) {
        outAtts.emplace_back(rightKey);
        projectionAtts.emplace_back(rightKey);
      }

      // make sure the input is not in the applied attributes because this would be bad
      it = std::find(inputAtts.begin(), inputAtts.end(), rightIn);
      if(it != inputAtts.end()) {
        throw runtime_error("Somehow the input was applied in the LHS and therefore we can not use the join-agg algorithm");
      }

      // add the computation
      outputRHS.emplace_back(comp);
    }

    /**
     * 3. Modify the join so that it keeps the key
     */

    auto &joinComp = *curr;

    // replace the inputs with keys
    if(joinComp->getOutput().getAtts().size() != 2){
      throw runtime_error("");
    }

    //
    for(auto &att : joinComp->getOutput().getAtts()) {

      // check if left input and change it to a key
      if(att == leftIn) {
        att = leftKey;
        continue;
      }

      // check if right input and change if to a key
      if(att == rightIn) {
        att = rightKey;
        continue;
      }
    }

    ((ApplyJoin*)joinComp.get())->getProjection().getAtts() = { leftKey };
    ((ApplyJoin*)joinComp.get())->getRightProjection().getAtts() = { rightKey };

    // add the computation
    outputRHS.emplace_back(joinComp);

    /**
     * 4. Now exclude any operator that is n
     */

    //
    auto inputSet = joinComp->getOutputName();

    // we store all the aliases for keys here
    std::set<std::string> lhsKeyAliases;
    std::set<std::string> rhsKeyAliases;
    std::set<std::string> attsToRemove;

    curr++;
    for(; curr != rightAtomicComputations.end(); curr++) {


      /**
       * 4.1.
       */

      // get the computation
      auto &comp = *curr;

      // if it is an aggergation break
      if(comp->getAtomicComputationTypeID() == ApplyAggTypeID) {
        break;
      }

      auto &outAtts = comp->getOutput().getAtts();
      auto &projectionAtts = comp->getProjection().getAtts();
      auto &inputAtts = comp->getInput().getAtts();


      /**
       * 4.2
       */

      //
      auto it = std::find(inputAtts.begin(), inputAtts.end(), leftIn);
      auto jt = std::find(inputAtts.begin(), inputAtts.end(), rightIn);
      bool toRemove = std::any_of(inputAtts.begin(), inputAtts.end(), [&](auto &it) { return attsToRemove.find(it) != attsToRemove.end(); });
      if(it != inputAtts.end() || jt != inputAtts.end() || toRemove) {

        // get the generated columns
        gen = (*curr)->getGeneratedColumns();

        // check if this is a key extraction
        kvPairs = (*curr)->getKeyValuePairs();
        kvIt = kvPairs->find("lambdaType");
        if(kvIt != kvPairs->end() && kvIt->second == "key" && (it != inputAtts.end() || jt != inputAtts.end())) {
          rhsKeyAliases.insert(gen.back());
        }
        // if it is not a key extraction that means that we have to remove all the
        else {

          // everything this tuple set generates into the attributes to remove
          attsToRemove.insert(gen.begin(), gen.end());
        }

        continue;
      }

      /**
       * 4.3 Update the output attributes
       */

      // replace all the inputs with the appropriate keys
      for(auto &att : outAtts) {

        // check if left input and change it to a key
        if(att == leftIn) {
          att = leftKey;
          continue;
        }

        // check if right input and change if to a key
        if(att == rightIn) {
          att = rightKey;
          continue;
        }
      }

      std::vector<std::string> tmp;
      for(auto &att : outAtts) {

        // ignore the attribute
        if(attsToRemove.find(att) != attsToRemove.end()) {
          continue;
        }

        // ignore if it is an alias
        if(lhsKeyAliases.find(att) != lhsKeyAliases.end()) {
          continue;
        }

        // ignore if it is an alias
        if(rhsKeyAliases.find(att) != rhsKeyAliases.end()) {
          continue;
        }

        // insert
        tmp.emplace_back(att);
      }
      outAtts = tmp;

      /**
       * 4.4 Update the input attributes
       */
      for(auto &att : inputAtts) {

        // replace the left key alias with the left key name
        if(lhsKeyAliases.find(att) != lhsKeyAliases.end()) {
          att = leftKey;
          continue;
        }

        // replace the right key alias with the right key name
        if(rhsKeyAliases.find(att) != rhsKeyAliases.end()) {
          att = rightKey;
          continue;
        }
      }

      /**
       * 4.x
       */

      // set the input
      comp->getProjection().getSetName() = inputSet;
      comp->getInput().getSetName() = inputSet;

      // the output of the current computation is becoming the next input set
      inputSet = comp->getOutputName();

      // add the computation
      outputRHS.emplace_back(comp);
    }

    // move the LHS
    rightAtomicComputations = std::move(outputRHS);
  }

  void transformLHS() {

    std::vector<AtomicComputationPtr> outputLHS;

    // get the scan set
    auto curr = leftAtomicComputations.begin();

    // get the input attribute
    leftIn = (*curr)->getOutput().getAtts().front();

    // go the the key extraction and make sure it is a key extraction
    curr++;
    auto &kvPairs = (*curr)->getKeyValuePairs();
    auto kvIt = kvPairs->find("lambdaType");
    if(kvIt == kvPairs->end() || kvIt->second != "key") {
      throw runtime_error("There is not key extraction lambda after the scan set.");
    }

    // get the key attribute there should either be (in, key) or (key, in)
    leftKey = (*curr)->getOutput().getAtts().front() == leftIn ? (*curr)->getOutput().getAtts().back() : (*curr)->getOutput().getAtts().front();

    //
    outputLHS.emplace_back(std::make_shared<ScanSet>(TupleSpec{(*curr)->getOutputName(), { leftKey }},
                                                     leftSet.first,
                                                     leftSet.second,
                                                     (*curr)->getComputationName()));

    curr++;
    for(; curr != leftAtomicComputations.end(); curr++) {

      auto &comp = *curr;

      auto &outAtts = comp->getOutput().getAtts();
      auto &projectionAtts = comp->getProjection().getAtts();
      auto &inputAtts = comp->getInput().getAtts();

      // remove the input
      auto it = std::find(outAtts.begin(), outAtts.end(), leftIn);
      outAtts.erase(it);

      // remove the from the input attributes
      it = std::find(projectionAtts.begin(), projectionAtts.end(), leftIn);
      projectionAtts.erase(it);

      // the key was removed add it
      it = std::find(outAtts.begin(), outAtts.end(), leftKey);
      if(it == outAtts.end()) {
        outAtts.emplace_back(leftKey);
        projectionAtts.emplace_back(leftKey);
      }

      // make sure the input is not in the applied attributes because this would be bad
      it = std::find(inputAtts.begin(), inputAtts.end(), leftIn);
      if(it != inputAtts.end()) {
        throw runtime_error("Somehow the input was applied in the LHS and therefore we can not use the join-agg algorithm");
      }

      // add the computation
      outputLHS.emplace_back(comp);
    }

    // move the LHS
    leftAtomicComputations = std::move(outputLHS);
  }

  std::string leftIn;
  std::string leftKey;

  std::string rightIn;
  std::string rightKey;

  std::vector<AtomicComputationPtr> leftAtomicComputations;
  std::vector<AtomicComputationPtr> rightAtomicComputations;

  std::string leftSide;
  std::string rightSide;
  std::string join;
  std::string agg;

  std::pair<std::string, std::string> leftSet;
  std::pair<std::string, std::string> rightSet;

  LogicalPlanPtr logicalPlan;

};

}

