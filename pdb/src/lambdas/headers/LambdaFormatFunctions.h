#pragma once

namespace pdb {

inline std::string formatAtomicComputation(const std::string &inputTupleSetName,
                                    const std::vector<std::string> &inputColumnNames,
                                    const std::vector<std::string> &inputColumnsToApply,
                                    const std::string &outputTupleSetName,
                                    const std::vector<std::string> &outputColumns,
                                    const std::string &outputColumnName,
                                    const std::string &tcapOperation,
                                    const std::string &computationNameAndLabel,
                                    const std::string &lambdaNameAndLabel,
                                    const std::map<std::string, std::string> &info) {

  mustache::mustache outputTupleSetNameTemplate
      {"{{outputTupleSetName}}({{#outputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/outputColumns}}) <= "
       "{{tcapOperation}} ({{inputTupleSetName}}({{#inputColumnsToApply}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumnsToApply}}), "
       "{{inputTupleSetName}}({{#hasColumnNames}}{{#inputColumnNames}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumnNames}}{{/hasColumnNames}}), "
       "'{{computationNameAndLabel}}', "
       "{{#hasLambdaNameAndLabel}}'{{lambdaNameAndLabel}}', {{/hasLambdaNameAndLabel}}"
       "[{{#info}}('{{key}}', '{{value}}'){{^isLast}}, {{/isLast}}{{/info}}])\n"};

  // create the data for the output columns
  mustache::data outputColumnData = mustache::from_vector<std::string>(outputColumns);

  // create the data for the input columns to apply
  mustache::data inputColumnsToApplyData = mustache::from_vector<std::string>(inputColumnsToApply);

  // create the data for the input columns to apply
  mustache::data inputColumnNamesData = mustache::from_vector<std::string>(inputColumnNames);

  // create the info data
  mustache::data infoData = mustache::from_map(info);

  // create the data for the lambda
  mustache::data lambdaData;

  lambdaData.set("outputTupleSetName", outputTupleSetName);
  lambdaData.set("outputColumns", outputColumnData);
  lambdaData.set("tcapOperation", tcapOperation);
  lambdaData.set("inputTupleSetName", inputTupleSetName);
  lambdaData.set("inputColumnsToApply", inputColumnsToApplyData);
  lambdaData.set("hasColumnNames", !inputColumnNames.empty());
  lambdaData.set("inputColumnNames", inputColumnNamesData);
  lambdaData.set("inputTupleSetName", inputTupleSetName);
  lambdaData.set("computationNameAndLabel", computationNameAndLabel);
  lambdaData.set("hasLambdaNameAndLabel", !lambdaNameAndLabel.empty());
  lambdaData.set("lambdaNameAndLabel", lambdaNameAndLabel);
  lambdaData.set("info", infoData);

  return outputTupleSetNameTemplate.render(lambdaData);
}

}