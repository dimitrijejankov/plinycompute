//
// Created by vicram on 3/26/19.
//

/**
 * This class is distinct from the AggregateCompBase in the master branch, but it serves
 * a similar purpose.
 */

#ifndef PDB_AGGREGATECOMPBASE_H
#define PDB_AGGREGATECOMPBASE_H

#include "Computation.h"

namespace pdb {

template<class OutputClass, class InputClass, class KeyClass, class ValueClass>
class AggregateCompBase : public Computation {

  /**
   * Gets the operation that extracts a key from an input object
   * @param aggMe - the object we want to get the operation from
   * @return the projection lambda. This must be a lambda which takes in
   * a Handle<InputClass> and constructs an object of type KeyClass.
   */
  virtual Lambda<KeyClass> getKeyProjection(Handle<InputClass> aggMe) = 0;

  /**
   * Gets the operation that extracts a value from an input object
   * @param aggMe - the object we want to get the operation from
   * @return the projection lambda. This must be a lambda which takes in
   * a Handle<InputClass> and constructs an object of type ValueClass.
   */
  virtual Lambda<ValueClass> getValueProjection(Handle<InputClass> aggMe) = 0;

  // extract the key projection and value projection
  void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal) override {
    int suffix = 0;
    Handle<InputClass> checkMe = nullptr;
    Lambda<KeyClass> keyLambda = getKeyProjection(checkMe);
    Lambda<ValueClass> valueLambda = getValueProjection(checkMe);
    keyLambda.toMap(returnVal, suffix);
    valueLambda.toMap(returnVal, suffix);
  }

  // this is an aggregation comp
  std::string getComputationType() override {
    return std::string("AggregationComp");
  }

  int getNumInputs() override {
    return 1;
  }

  // gets the name of the i^th input type...
  std::string getIthInputType(int i) override {
    if (i == 0) {
      return getTypeName<InputClass>();
    } else {
      return "";
    }
  }

  // gets the output type of this query as a string
  std::string getOutputType() override {
    return getTypeName<OutputClass>();
  }

  // below function implements the interface for parsing computation into a TCAP string
  std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets,
                           int computationLabel,
                           std::string &outputTupleSetName,
                           std::vector<std::string> &outputColumnNames,
                           std::string &addedOutputColumnName) override {

    if (inputTupleSets.empty()) {
      return "";
    }

    InputTupleSetSpecifier inputTupleSet = inputTupleSets[0];
    return toTCAPString(inputTupleSet.getTupleSetName(),
                        inputTupleSet.getColumnNamesToKeep(),
                        inputTupleSet.getColumnNamesToApply(),
                        computationLabel,
                        outputTupleSetName,
                        outputColumnNames,
                        addedOutputColumnName);
  }

  // to return Aggregate tcap string
  std::string toTCAPString(std::string inputTupleSetName,
                           std::vector<std::string> inputColumnNames,
                           std::vector<std::string> inputColumnsToApply,
                           int computationLabel,
                           std::string &outputTupleSetName,
                           std::vector<std::string> &outputColumnNames,
                           std::string &addedOutputColumnName) {
    return "";
  }

  /**
   * Return the hash sink for the aggregation
   * @param consumeMe -
   * @param projection
   * @param plan
   * @return
   */
  ComputeSinkPtr getComputeSink(TupleSpec &consumeMe, TupleSpec &projection) override {
    return std::make_shared<pdb::HashSink<KeyClass, ValueClass>>(consumeMe, projection);
  }

  ComputeSourcePtr getComputeSource(const PDBAbstractPageSetPtr &pageSet, size_t chunkSize, uint64_t workerID) override {
    return std::make_shared<MapTupleSetIterator<KeyClass, ValueClass, OutputClass>> (pageSet, workerID, chunkSize);
  }

  bool needsMaterializeOutput() override {
    return false;
  }
};

}

#endif //PDB_AGGREGATECOMPBASE_H
