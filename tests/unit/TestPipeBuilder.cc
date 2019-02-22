//
// Created by dimitrije on 2/22/19.
//

#include <gtest/gtest.h>
#include <AtomicComputationList.h>
#include <Parser.h>
#include <PDBPipeNodeBuilder.h>

TEST(BufferManagerBackendTest, Test1) {


  std::string myLogicalPlan =
      "inputDataForScanSet_0(in0) <= SCAN ('input_set', 'by8_db', 'ScanSet_0') \n"\
      "nativ_0OutForSelectionComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForScanSet_0(in0), inputDataForScanSet_0(in0), 'SelectionComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')]) \n"\
      "filteredInputForSelectionComp1(in0) <= FILTER (nativ_0OutForSelectionComp1(nativ_0_1OutFor), nativ_0OutForSelectionComp1(in0), 'SelectionComp_1') \n"\
      "nativ_1OutForSelectionComp1 (nativ_1_1OutFor) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(), 'SelectionComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')]) \n"\
      "nativ_1OutForSelectionComp1_out( ) <= OUTPUT ( nativ_1OutForSelectionComp1 ( nativ_1_1OutFor ), 'output_set', 'by8_db', 'SetWriter_2') \n";

  // get the string to compile
  myLogicalPlan.push_back('\0');

  // where the result of the parse goes
  AtomicComputationList *myResult;

  // now, do the compilation
  yyscan_t scanner;
  LexerExtra extra{""};
  yylex_init_extra(&extra, &scanner);
  const YY_BUFFER_STATE buffer{yy_scan_string(myLogicalPlan.data(), scanner)};
  const int parseFailed{yyparse(scanner, &myResult)};
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  // if it didn't parse, get outta here
  if (parseFailed) {
    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
    exit(1);
  }

  // this is the logical plan to return
  auto atomicComputations = std::shared_ptr<AtomicComputationList>(myResult);

  pdb::PDBPipeNodeBuilder factory(atomicComputations);

  auto out = factory.generateAnalyzerGraph();
}

TEST(BufferManagerBackendTest, Test2) {

  std::string myLogicalPlan =
      "inputData (in) <= SCAN ('mySet', 'myData', 'ScanSet_0', []) \n\
       inputWithAtt (in, att) <= APPLY (inputData (in), inputData (in), 'SelectionComp_1', 'methodCall_0', []) \n\
       inputWithAttAndMethod (in, att, method) <= APPLY (inputWithAtt (in), inputWithAtt (in, att), 'SelectionComp_1', 'attAccess_1', []) \n\
       inputWithBool (in, bool) <= APPLY (inputWithAttAndMethod (att, method), inputWithAttAndMethod (in), 'SelectionComp_1', '==_2', []) \n\
       filteredInput (in) <= FILTER (inputWithBool (bool), inputWithBool (in), 'SelectionComp_1', []) \n\
       projectedInputWithPtr (out) <= APPLY (filteredInput (in), filteredInput (), 'SelectionComp_1', 'methodCall_3', []) \n\
       projectedInput (out) <= APPLY (projectedInputWithPtr (out), projectedInputWithPtr (), 'SelectionComp_1', 'deref_4', []) \n\
       aggWithKeyWithPtr (out, key) <= APPLY (projectedInput (out), projectedInput (out), 'AggregationComp_2', 'attAccess_0', []) \n\
       aggWithKey (out, key) <= APPLY (aggWithKeyWithPtr (key), aggWithKeyWithPtr (out), 'AggregationComp_2', 'deref_1', []) \n\
       aggWithValue (key, value) <= APPLY (aggWithKey (out), aggWithKey (key), 'AggregationComp_2', 'methodCall_2', []) \n\
       agg (aggOut) <=	AGGREGATE (aggWithValue (key, value), 'AggregationComp_2', []) \n\
       checkSales (aggOut, isSales) <= APPLY (agg (aggOut), agg (aggOut), 'SelectionComp_3', 'methodCall_0', []) \n\
       justSales (aggOut, isSales) <= FILTER (checkSales (isSales), checkSales (aggOut), 'SelectionComp_3', []) \n\
       final (result) <= APPLY (justSales (aggOut), justSales (), 'SelectionComp_3', 'methodCall_1', []) \n\
	   nothing () <= OUTPUT (final (result), 'outSet', 'myDB', 'SetWriter_4', [])";

  // get the string to compile
  myLogicalPlan.push_back('\0');

  // where the result of the parse goes
  AtomicComputationList *myResult;

  // now, do the compilation
  yyscan_t scanner;
  LexerExtra extra{""};
  yylex_init_extra(&extra, &scanner);
  const YY_BUFFER_STATE buffer{yy_scan_string(myLogicalPlan.data(), scanner)};
  const int parseFailed{yyparse(scanner, &myResult)};
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  // if it didn't parse, get outta here
  if (parseFailed) {
    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
    exit(1);
  }

  // this is the logical plan to return
  auto atomicComputations = std::shared_ptr<AtomicComputationList>(myResult);

  pdb::PDBPipeNodeBuilder factory(atomicComputations);

  auto out = factory.generateAnalyzerGraph();
}