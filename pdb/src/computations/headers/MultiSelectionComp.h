#pragma once

#include "Computation.h"
#include "ComputePlan.h"
#include "VectorSink.h"
#include "TypeName.h"
#include "SetScanner.h"

namespace pdb {

/**
 * TODO add proper description
 * @tparam Out
 * @tparam InputClass
 */
template<class Out, class Input>
class MultiSelectionComp : public Computation {

 public:

  /**
   * the computation returned by this method is called to see if a data item should be returned in the output set
   * @param checkMe
   * @return
   */
  virtual pdb::Lambda<bool> getSelection(pdb::Handle<Input> checkMe) = 0;

  /**
   * the computation returned by this method is called to produce output tuples from this method
   * @param checkMe
   * @return
   */
  virtual pdb::Lambda<pdb::Vector<pdb::Handle<Out>>> getProjection(pdb::Handle<Input> checkMe) = 0;

  /**
   * calls getProjection and getSelection to extract the lambdas
   * @param returnVal
   */
  void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal) override {
    Handle<Input> checkMe = nullptr;
    Lambda<bool> selectionLambda = getSelection(checkMe);
    Lambda<Vector<Handle<Out>>> projectionLambda = getProjection(checkMe);

    // the label we are started labeling
    int32_t startLabel = 0;

    // extract the lambdas
    selectionLambda.extractLambdas(returnVal, startLabel);
    projectionLambda.extractLambdas(returnVal, startLabel);
  }

  /**
   * this is a MultiSelection computation
   * @return
   */
  std::string getComputationType() override {
    return std::string("MultiSelectionComp");
  }

  /**
   * gets the name of the i^th input type...
   * @param i
   * @return
   */
  std::string getInputType(int i) override {
    if (i == 0) {
      return getTypeName<Input>();
    } else {
      return "";
    }
  }

  /**
   * get the number of inputs to this query type
   * @return
   */
  int getNumInputs() override {
    return 1;
  }

  /**
   * return the output type
   * @return
   */
  std::string getOutputType() override {
    return getTypeName<Out>();
  }

  pdb::ComputeSourcePtr getComputeSource(const PDBAbstractPageSetPtr &pageSet,
                                         size_t chunkSize,
                                         uint64_t workerID,
                                         std::map<ComputeInfoType, ComputeInfoPtr> &) override {
    return std::make_shared<pdb::VectorTupleSetIterator>(pageSet, chunkSize, workerID);
  }

  pdb::ComputeSinkPtr getComputeSink(TupleSpec &consumeMe, TupleSpec &, TupleSpec &projection, uint64_t,
                                     std::map<ComputeInfoType, ComputeInfoPtr> &, pdb::LogicalPlanPtr &) override {
    return std::make_shared<pdb::VectorSink<Out>>(consumeMe, projection);
  }

  /**
   * below function implements the interface for parsing computation into a TCAP string
   * @param inputTupleSets
   * @param computationLabel
   * @param outputTupleSetName
   * @param outputColumnNames
   * @param addedOutputColumnName
   * @return
   */
  std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets,
                           int computationLabel) override {

    if (inputTupleSets.empty()) {
      return "";
    }

    InputTupleSetSpecifier inputTupleSet = inputTupleSets[0];

    // make the inputs
    std::string inputTupleSetName = inputTupleSet.getTupleSetName();
    std::vector<std::string> inputColumnNames = inputTupleSet.getColumnNamesToKeep();
    std::vector<std::string> inputColumnsToApply = inputTupleSet.getColumnNamesToApply();

    /**
     * 1. Generate the TCAP for the selection predicate
     */

    // this is going to have info about the input
    assert(inputTupleSets.size() == 1);
    MultiInputsBase multiInputsBase(1);

    // set the name of the tuple set for the i-th position
    multiInputsBase.tupleSetNamesForInputs[0] = inputTupleSets[0].getTupleSetName();

    // set the columns for the i-th position
    multiInputsBase.inputColumnsForInputs[0] = inputTupleSets[0].getColumnNamesToKeep();

    // the the columns to apply for the i-th position
    multiInputsBase.inputColumnsToApplyForInputs[0] = inputTupleSets[0].getColumnNamesToApply();

    // setup all input names (the column name corresponding to input in tuple set) has to be one
    assert(inputTupleSets[0].getColumnNamesToApply().size() == 1);
    multiInputsBase.inputNames[0] = inputTupleSets[0].getColumnNamesToApply()[0];

    // we want to keep the input, so that it can be used by the projection
    multiInputsBase.inputColumnsToKeep = { inputTupleSets[0].getColumnNamesToKeep()[0] };

    // we label the lambdas within this computation from zero
    int lambdaLabel = 0;

    // call the selection
    GenericHandle checkMe (1);
    Lambda<bool> selectionLambda = getSelection(checkMe);

    // create multi selection
    std::string tcapString;
    tcapString += "\n/* Apply MultiSelection filtering */\n";
    tcapString += selectionLambda.toTCAPString(lambdaLabel,
                                               getComputationType(),
                                               computationLabel,
                                               false,
                                               &multiInputsBase);

    // get the columns for the TCAP
    auto appliedColumns = multiInputsBase.inputColumnsToApplyForInputs[0];
    auto outputColumns = multiInputsBase.getNotAppliedInputColumnsForIthInput(0);
    auto tupleSetName = multiInputsBase.tupleSetNamesForInputs[0];

    // make sure there is only one output column
    assert(outputColumns.size() == 1);

    // create the data for the column names
    mustache::data inputColumnsToApplyData = mustache::from_vector<std::string>(multiInputsBase.inputColumnsToApplyForInputs[0]);
    mustache::data outputColumnsData = mustache::from_vector<std::string>(multiInputsBase.getNotAppliedInputColumnsForIthInput(0));

    // create the data for the filter
    mustache::data selectionCompData;
    selectionCompData.set("computationType", getComputationType());
    selectionCompData.set("computationLabel", std::to_string(computationLabel));
    selectionCompData.set("inputColumns", inputColumnsToApplyData);
    selectionCompData.set("outputColumns", outputColumnsData);
    selectionCompData.set("tupleSetName", tupleSetName);

    // tupleSetName1(att1, att2, ...) <= FILTER (tupleSetName(methodCall_0OutFor_isFrank), methodCall_0OutFor_SelectionComp1(in0), 'SelectionComp_1')
    mustache::mustache scanSetTemplate
        {"filteredInputFor{{computationType}}{{computationLabel}}({{#outputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/outputColumns}}) "
         "<= FILTER ({{tupleSetName}}({{#inputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumns}}), "
         "{{tupleSetName}}({{#outputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/outputColumns}}), '{{computationType}}_{{computationLabel}}')\n"};

    // generate the TCAP string for the FILTER
    tcapString += scanSetTemplate.render(selectionCompData);

    /**
     * 2. Generate the TCAP for the projection
     */

    // generate the new tuple set name
    mustache::mustache newTupleSetNameTemplate{"filteredInputFor{{computationType}}{{computationLabel}}"};
    multiInputsBase.tupleSetNamesForInputs[0] = newTupleSetNameTemplate.render(selectionCompData);

    Lambda<Vector<Handle<Out>>> projectionLambda = getProjection(checkMe);
    tcapString += "\n/* Apply MultiSelection projection */\n";
    tcapString += projectionLambda.toTCAPString(lambdaLabel,
                                                getComputationType(),
                                                computationLabel,
                                                true,
                                                &multiInputsBase);

    /**
     * 3. Generate the FLATTEN operator
     */

    assert(multiInputsBase.inputColumnsToApplyForInputs.size() == 1);
    assert(multiInputsBase.tupleSetNamesForInputs.size() == 1);
    assert(multiInputsBase.inputColumnsToApplyForInputs[0].size() == 1);

    // add the new data
    selectionCompData.set("inputColumn", multiInputsBase.inputColumnsToApplyForInputs[0][0]);
    selectionCompData.set("computationType", getComputationType());
    selectionCompData.set("computationLabel", std::to_string(computationLabel));
    selectionCompData.set("inputTupleSetName", multiInputsBase.tupleSetNamesForInputs[0]);

    // create the new tuple set name
    newTupleSetNameTemplate = {"flattenedOutFor{{computationType}}{{computationLabel}}"};
    outputTupleSetName = newTupleSetNameTemplate.render(selectionCompData);

    // create the new output column name
    mustache::mustache newOutputColumnNameTemplate = {"flattened_{{inputColumn}}"};
    outputColumnToApply = newOutputColumnNameTemplate.render(selectionCompData);

    // add flatten
    mustache::mustache flattenTemplate{"flattenedOutFor{{computationType}}{{computationLabel}}(flattened_{{inputColumn}})"
                                       " <= FLATTEN ({{inputTupleSetName}}({{inputColumn}}), "
                                       "{{inputTupleSetName}}(), '{{computationType}}_{{computationLabel}}')\n"};
    tcapString += flattenTemplate.render(selectionCompData);

    // mark this as traversed
    this->traversed = true;

    // return the tcap
    return std::move(tcapString);
  }

};

}