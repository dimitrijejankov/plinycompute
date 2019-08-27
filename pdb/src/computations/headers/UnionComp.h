#pragma once

#include "UnionCompBase.h"

namespace pdb {

template<typename Derived, typename InOut>
class UnionComp : public UnionCompBase {
public:

  // return the output type
  std::string getOutputType() override {
    return getTypeName<InOut>();
  }

  // count the number of inputs
  int getNumInputs() final {
    return 2;
  }

  string getInputType(int i) override {
    return getTypeName<InOut>();
  }

  std::string getComputationType() override {
    return std::string("UnionComp");
  }

  std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets, int computationLabel) override {

    if (inputTupleSets.size() != getNumInputs()) {
      std::cout << "ERROR: inputTupleSet size is " << inputTupleSets.size() << " and not equivalent with Join's inputs "
                << getNumInputs() << std::endl;
      return "";
    }

    // create the data for the filter
    mustache::data data;
    data.set("computationType", getComputationType());
    data.set("computationLabel", std::to_string(computationLabel));

    data.set("leftTupleSetName", inputTupleSets[0].getTupleSetName());
    data.set("rightTupleSetName", inputTupleSets[1].getTupleSetName());

    // set the lhs input columns
    mustache::data lhsColumns = mustache::from_vector<std::string>(inputTupleSets[0].getColumnNamesToApply());
    data.set("leftColumns", lhsColumns);

    // set the rhs input columns
    mustache::data rhsColumns = mustache::from_vector<std::string>(inputTupleSets[1].getColumnNamesToApply());
    data.set("rightColumns", rhsColumns);

    // set the output columns
    mustache::mustache newAddedOutputColumnName1Template{"unionOutFor{{computationLabel}}"};
    std::string outputColumn = newAddedOutputColumnName1Template.render(data);
    data.set("outputColumn", outputColumn);

    // the union template
    mustache::mustache unionTemplate{"unionOut{{computationType}}{{computationLabel}} ({{outputColumn}})"
                                    "<= UNION ({{leftTupleSetName}}({{#leftColumns}}{{value}}{{^isLast}},{{/isLast}}{{/leftColumns}}),"
                                              "{{rightTupleSetName}}({{#rightColumns}}{{value}}{{^isLast}},{{/isLast}}{{/rightColumns}}),"
                                              "'{{computationType}}_{{computationLabel}}')\n"};


    std::string tcapString = std::move(unionTemplate.render(data));

    // update the state of the computation
    mustache::mustache newTupleSetNameTemplate{"unionOut{{computationType}}{{computationLabel}}"};
    outputTupleSetName = newTupleSetNameTemplate.render(data);

    // set the output column
    this->outputColumnToApply = outputColumn;

    // update marker
    this->traversed = true;

    // return the tcap
    return std::move(tcapString);
  }

};

}