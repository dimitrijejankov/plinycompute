#pragma once

namespace pdb {

inline std::string formatLambdaComputation(const std::string &inputTupleSetName,
                                           const std::vector<std::string> &inputColumnNames,
                                           const std::vector<std::string> &inputColumnsToApply,
                                           const std::string &outputTupleSetName,
                                           const std::vector<std::string> &outputColumns,
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
  lambdaData.set("computationNameAndLabel", computationNameAndLabel);
  lambdaData.set("hasLambdaNameAndLabel", !lambdaNameAndLabel.empty());
  lambdaData.set("lambdaNameAndLabel", lambdaNameAndLabel);
  lambdaData.set("info", infoData);

  return outputTupleSetNameTemplate.render(lambdaData);
}

inline std::string formatJoinComputation(const std::string &outputTupleSetName,
                                         const std::vector<std::string> &outputColumns,
                                         const std::string &lhsInputTupleSetName,
                                         const std::vector<std::string> &lhsInputColumnsToApply,
                                         const std::vector<std::string> &lhsInputColumnNames,
                                         const std::string &rhsInputTupleSetName,
                                         const std::vector<std::string> &rhsInputColumnsToApply,
                                         const std::vector<std::string> &rhsInputColumnNames,
                                         const std::string &computationNameAndLabel) {

  mustache::mustache outputTupleSetNameTemplate
  {"{{outputTupleSetName}}({{#outputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/outputColumns}}) <= "
   "JOIN ({{lhsInputTupleSetName}}({{#lhsInputColumnsToApply}}{{value}}{{^isLast}},{{/isLast}}{{/lhsInputColumnsToApply}}), "
         "{{lhsInputTupleSetName}}({{#lhsInputColumnNames}}{{value}}{{^isLast}},{{/isLast}}{{/lhsInputColumnNames}}), "
         "{{rhsInputTupleSetName}}({{#rhsInputColumnsToApply}}{{value}}{{^isLast}},{{/isLast}}{{/rhsInputColumnsToApply}}), "
         "{{rhsInputTupleSetName}}({{#rhsInputColumnNames}}{{value}}{{^isLast}},{{/isLast}}{{/rhsInputColumnNames}}), "
         "'{{computationNameAndLabel}}')\n"};

  // create the data for the output columns
  mustache::data outputColumnData = mustache::from_vector<std::string>(outputColumns);

  // create the data for the input columns to apply
  mustache::data lhsInputColumnsToApplyData = mustache::from_vector<std::string>(lhsInputColumnsToApply);

  // create the data for the input columns to apply
  mustache::data lhsInputColumnNamesData = mustache::from_vector<std::string>(lhsInputColumnNames);

  // create the data for the input columns to apply
  mustache::data rhsInputColumnsToApplyData = mustache::from_vector<std::string>(rhsInputColumnsToApply);

  // create the data for the input columns to apply
  mustache::data rhsInputColumnNamesData = mustache::from_vector<std::string>(rhsInputColumnNames);

  // create the data for the lambda
  mustache::data lambdaData;

  // set the lambda data
  lambdaData.set("outputTupleSetName", outputTupleSetName);
  lambdaData.set("outputColumns", outputColumnData);
  lambdaData.set("lhsInputTupleSetName", lhsInputTupleSetName);
  lambdaData.set("lhsInputColumnNames", lhsInputColumnNamesData);
  lambdaData.set("lhsInputColumnsToApply", lhsInputColumnsToApplyData);
  lambdaData.set("rhsInputTupleSetName", rhsInputTupleSetName);
  lambdaData.set("rhsInputColumnNames", rhsInputColumnNamesData);
  lambdaData.set("rhsInputColumnsToApply", rhsInputColumnsToApplyData);
  lambdaData.set("computationNameAndLabel", computationNameAndLabel);

  // render it
  return outputTupleSetNameTemplate.render(lambdaData);
}


inline std::string formatFilterComputation(const std::string &outputTupleSetName,
                                           const std::vector<std::string> &outputColumns,
                                           const std::string &inputTupleSetName,
                                           const std::vector<std::string> &inputColumnsToApply,
                                           const std::vector<std::string> &inputColumnNames,
                                           const std::string &computationNameAndLabel) {


  mustache::mustache outputTupleSetNameTemplate
  {"{{outputTupleSetName}}({{#outputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/outputColumns}}) <= "
   "FILTER ({{inputTupleSetName}}({{#inputColumnsToApply}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumnsToApply}}), "
           "{{inputTupleSetName}}({{#inputColumnNames}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumnNames}}), "
           "'JoinComp_2')\n"};

  // create the data for the output columns
  mustache::data outputColumnData = mustache::from_vector<std::string>(outputColumns);

  // create the data for the input columns to apply
  mustache::data inputColumnsToApplyData = mustache::from_vector<std::string>(inputColumnsToApply);

  // create the data for the input columns to apply
  mustache::data inputColumnNamesData = mustache::from_vector<std::string>(inputColumnNames);

  // create the data for the lambda
  mustache::data lambdaData;

  // fill out the data
  lambdaData.set("outputTupleSetName", outputTupleSetName);
  lambdaData.set("outputColumns", outputColumnData);
  lambdaData.set("inputTupleSetName", inputTupleSetName);
  lambdaData.set("inputColumnsToApply", inputColumnsToApplyData);
  lambdaData.set("inputColumnNames", inputColumnNamesData);
  lambdaData.set("computationNameAndLabel", computationNameAndLabel);

  // output the tuple set
  return outputTupleSetNameTemplate.render(lambdaData);
}

}