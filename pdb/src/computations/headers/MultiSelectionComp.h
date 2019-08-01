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
    int suffix = 0;
    Handle<Input> checkMe = nullptr;
    Lambda<bool> selectionLambda = getSelection(checkMe);
    Lambda<Vector<Handle<Out>>> projectionLambda = getProjection(checkMe);
    selectionLambda.toMap(returnVal, suffix);
    projectionLambda.toMap(returnVal, suffix);
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
    std::vector<std::string> childrenLambdaNames;
    std::string myLambdaName;

    // make the inputs
    std::string inputTupleSetName = inputTupleSet.getTupleSetName();
    std::vector<std::string> inputColumnNames = inputTupleSet.getColumnNamesToKeep();
    std::vector<std::string> inputColumnsToApply = inputTupleSet.getColumnNamesToApply();

    // this is going to have info about the inputs
    MultiInputsBase multiInputsBase(this->getNumInputs());

    // update tuple set name for input sets
    for (unsigned int i = 0; i < inputTupleSets.size(); i++) {

      // set the name of the tuple set for the i-th position
      multiInputsBase.setTupleSetNameForIthInput(i, inputTupleSets[i].getTupleSetName());

      // set the columns for the i-th position
      multiInputsBase.setInputColumnsForIthInput(i, inputTupleSets[i].getColumnNamesToKeep());

      // the the columns to apply for the i-th position
      multiInputsBase.setInputColumnsToApplyForIthInput(i, inputTupleSets[i].getColumnNamesToApply());

      // setup all input names (the column name corresponding to input in tuple set)
      multiInputsBase.setNameForIthInput(i, inputTupleSets[i].getColumnNamesToApply()[0]);
    }

    PDB_COUT << "To GET TCAP STRING FOR SELECTION" << std::endl;

    GenericHandle checkMe (1);
    PDB_COUT << "TO GET TCAP STRING FOR SELECTION LAMBDA" << std::endl;
    Lambda<bool> selectionLambda = getSelection(checkMe);
    std::string tupleSetName;
    std::vector<std::string> columnNames;
    std::string addedColumnName;
    int lambdaLabel = 0;

    std::string tcapString;
    tcapString += "\n/* Apply MultiSelection filtering */\n";
    tcapString += selectionLambda.toTCAPString(childrenLambdaNames,
                                               lambdaLabel,
                                               getComputationType(),
                                               computationLabel,
                                               myLambdaName,
                                               false,
                                               &multiInputsBase);

    PDB_COUT << "tcapString after parsing selection lambda: " << tcapString << std::endl;
    PDB_COUT << "lambdaLabel=" << lambdaLabel << std::endl;

    // create the data for the column names
    mustache::data inputColumnData = mustache::data::type::list;
    for (int i = 0; i < multiInputsBase.getInputColumnsForIthInput(0).size(); i++) {

      mustache::data columnData;

      // fill in the column data
      columnData.set("columnName", multiInputsBase.getInputColumnsForIthInput(0)[i]);
      columnData.set("isLast", i == multiInputsBase.getInputColumnsForIthInput(0).size() - 1);

      inputColumnData.push_back(columnData);
    }

    // create the data for the filter
    mustache::data selectionCompData;
    selectionCompData.set("computationType", getComputationType());
    selectionCompData.set("computationLabel", std::to_string(computationLabel));
    selectionCompData.set("inputColumns", inputColumnData);
    selectionCompData.set("tupleSetName", tupleSetName);
    selectionCompData.set("addedColumnName", addedColumnName);

    // set the new tuple set name
    mustache::mustache newTupleSetNameTemplate{"filteredInputFor{{computationType}}{{computationLabel}}"};

    mustache::mustache filterTemplate{"filteredInputFor{{computationType}}{{computationLabel}}"
                                      "({{#inputColumns}}{{columnName}}{{^isLast}}, {{/isLast}}{{/inputColumns}}) "
                                      "<= FILTER ({{tupleSetName}}({{addedColumnName}}), {{tupleSetName}}"
                                      "({{#inputColumns}}{{columnName}}{{^isLast}}, {{/isLast}}{{/inputColumns}}), "
                                      "'{{computationType}}_{{computationLabel}}')\n"};

    tcapString += filterTemplate.render(selectionCompData);

    PDB_COUT << "tcapString after adding filter operation: " << tcapString << std::endl;
    PDB_COUT << "TO GET TCAP STRING FOR PROJECTION LAMBDA" << std::endl;
    PDB_COUT << "lambdaLabel=" << lambdaLabel << std::endl;

    // generate the new tuple set name
    multiInputsBase.setTupleSetNameForIthInput(0, newTupleSetNameTemplate.render(selectionCompData));

    // TODO make this nicer
    std::string outputTupleSetName;
    std::vector<std::string> outputColumnNames;
    std::string addedOutputColumnName;

    std::string newTupleSetName;

    Lambda<Vector<Handle<Out>>> projectionLambda = getProjection(checkMe);
    tcapString += "\n/* Apply MultiSelection projection */\n";
    tcapString += projectionLambda.toTCAPString(childrenLambdaNames,
                                                lambdaLabel,
                                                getComputationType(),
                                                computationLabel,
                                                myLambdaName,
                                                true,
                                                &multiInputsBase);

    // add the new data
    selectionCompData.set("addedOutputColumnName", addedOutputColumnName);
    selectionCompData.set("computationType", getComputationType());
    selectionCompData.set("computationLabel", std::to_string(computationLabel));
    selectionCompData.set("outputTupleSetName", outputTupleSetName);


    // create the new tuple set name
    newTupleSetNameTemplate = {"flattenedOutFor{{computationType}}{{computationLabel}}"};
    newTupleSetName = newTupleSetNameTemplate.render(selectionCompData);

    // create the new output column name
    mustache::mustache newOutputColumnNameTemplate = {"flattened_{{addedOutputColumnName}}"};
    std::string newOutputColumnName = newOutputColumnNameTemplate.render(selectionCompData);

    // add flatten
    mustache::mustache flattenTemplate{"flattenedOutFor{{computationType}}{{computationLabel}}(flattened_{{addedOutputColumnName}})"
                                       " <= FLATTEN ({{outputTupleSetName}}({{addedOutputColumnName}}), "
                                       "{{outputTupleSetName}}(), '{{computationType}}_{{computationLabel}}')\n"};
    tcapString += flattenTemplate.render(selectionCompData);

    this->traversed = true;
    this->outputTupleSetName = newTupleSetName;
    this->outputColumnToApply = newOutputColumnName;
    addedOutputColumnName = newOutputColumnName;
    outputColumnNames.clear();
    outputColumnNames.push_back(addedOutputColumnName);

    return tcapString;
  }

};

}