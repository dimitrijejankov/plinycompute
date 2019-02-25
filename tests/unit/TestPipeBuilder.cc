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

  EXPECT_EQ(out.size(), 1);

  int i = 0;
  for(auto &it : out.front()->getPipeComputations()) {

    switch (i) {

      case 0: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
        EXPECT_EQ(it->getOutputName(), "inputDataForScanSet_0");

        break;
      };
      case 1: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "nativ_0OutForSelectionComp1");

        break;
      };
      case 2: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
        EXPECT_EQ(it->getOutputName(), "filteredInputForSelectionComp1");

        break;
      };
      case 3: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "nativ_1OutForSelectionComp1");

        break;
      };
      case 4: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), WriteSetTypeID);
        EXPECT_EQ(it->getOutputName(), "nativ_1OutForSelectionComp1_out");

        break;
      };
      default: { EXPECT_FALSE(true); break;};
    }

    // increment
    i++;
  }

}

TEST(BufferManagerBackendTest, Test2) {

  std::string myLogicalPlan =
      "inputData (in) <= SCAN ('mySet', 'myData', 'ScanSet_0', []) \n"\
      "inputWithAtt (in, att) <= APPLY (inputData (in), inputData (in), 'SelectionComp_1', 'methodCall_0', []) \n"\
      "inputWithAttAndMethod (in, att, method) <= APPLY (inputWithAtt (in), inputWithAtt (in, att), 'SelectionComp_1', 'attAccess_1', []) \n"\
      "inputWithBool (in, bool) <= APPLY (inputWithAttAndMethod (att, method), inputWithAttAndMethod (in), 'SelectionComp_1', '==_2', []) \n"\
      "filteredInput (in) <= FILTER (inputWithBool (bool), inputWithBool (in), 'SelectionComp_1', []) \n"\
      "projectedInputWithPtr (out) <= APPLY (filteredInput (in), filteredInput (), 'SelectionComp_1', 'methodCall_3', []) \n"\
      "projectedInput (out) <= APPLY (projectedInputWithPtr (out), projectedInputWithPtr (), 'SelectionComp_1', 'deref_4', []) \n"\
      "aggWithKeyWithPtr (out, key) <= APPLY (projectedInput (out), projectedInput (out), 'AggregationComp_2', 'attAccess_0', []) \n"\
      "aggWithKey (out, key) <= APPLY (aggWithKeyWithPtr (key), aggWithKeyWithPtr (out), 'AggregationComp_2', 'deref_1', []) \n"\
      "aggWithValue (key, value) <= APPLY (aggWithKey (out), aggWithKey (key), 'AggregationComp_2', 'methodCall_2', []) \n"\
      "agg (aggOut) <=	AGGREGATE (aggWithValue (key, value), 'AggregationComp_2', []) \n"\
      "checkSales (aggOut, isSales) <= APPLY (agg (aggOut), agg (aggOut), 'SelectionComp_3', 'methodCall_0', []) \n"\
      "justSales (aggOut, isSales) <= FILTER (checkSales (isSales), checkSales (aggOut), 'SelectionComp_3', []) \n"\
      "final (result) <= APPLY (justSales (aggOut), justSales (), 'SelectionComp_3', 'methodCall_1', []) \n"\
	  "nothing () <= OUTPUT (final (result), 'outSet', 'myDB', 'SetWriter_4', [])";

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

  EXPECT_EQ(out.size(), 1);

  auto c = out.front();
  int i = 0;
  for(auto &it : c->getPipeComputations()) {

    switch (i) {

      case 0: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
        EXPECT_EQ(it->getOutputName(), "inputData");

        break;
      };
      case 1: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "inputWithAtt");

        break;
      };
      case 2: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "inputWithAttAndMethod");

        break;
      };
      case 3: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "inputWithBool");

        break;
      };
      case 4: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
        EXPECT_EQ(it->getOutputName(), "filteredInput");

        break;
      };
      case 5: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "projectedInputWithPtr");

        break;
      };
      case 6: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "projectedInput");

        break;
      };
      case 7: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "aggWithKeyWithPtr");

        break;
      };
      case 8: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "aggWithKey");

        break;
      };
      case 9: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "aggWithValue");

        break;
      };
      default: { EXPECT_FALSE(true); break;};
    }

    // increment
    i++;
  }

  auto producers = c->getConsumers();
  EXPECT_EQ(producers.size(), 1);

  i = 0;
  for(auto &it : producers.front()->getPipeComputations()) {
    switch (i) {

      case 0: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyAggTypeID);
        EXPECT_EQ(it->getOutputName(), "agg");

        break;
      };
      case 1: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "checkSales");

        break;
      };
      case 2: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
        EXPECT_EQ(it->getOutputName(), "justSales");

        break;
      };
      case 3: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
        EXPECT_EQ(it->getOutputName(), "final");

        break;
      };
      case 4: {

        EXPECT_EQ(it->getAtomicComputationTypeID(), WriteSetTypeID);
        EXPECT_EQ(it->getOutputName(), "nothing");

        break;
      };
      default: { EXPECT_FALSE(true); break;};
    }

    i++;
  }

}

TEST(BufferManagerBackendTest, Test3) {
  std::string myLogicalPlan = "/* scan the three inputs */ \n"\
	  "A (a) <= SCAN ('mySet', 'myData', 'ScanSet_0', []) \n"\
	  "B (aAndC) <= SCAN ('mySet', 'myData', 'ScanSet_1', []) \n"\
	  "C (c) <= SCAN ('mySet', 'myData', 'ScanSet_2', []) \n"\
	  "\n"\
      "/* extract and hash a from A */ \n"\
      "AWithAExtracted (a, aExtracted) <= APPLY (A (a), A(a), 'JoinComp_3', 'self_0', []) \n"\
      "AHashed (a, hash) <= HASHLEFT (AWithAExtracted (aExtracted), A (a), 'JoinComp_3', '==_2', []) \n"\
      "\n"\
      "/* extract and hash a from B */ \n"\
      "BWithAExtracted (aAndC, a) <= APPLY (B (aAndC), B (aAndC), 'JoinComp_3', 'attAccess_1', []) \n"\
      "BHashedOnA (aAndC, hash) <= HASHRIGHT (BWithAExtracted (a), BWithAExtracted (aAndC), 'JoinComp_3', '==_2', []) \n"\
      "\n"\
      "/* now, join the two of them */ \n"\
      "AandBJoined (a, aAndC) <= JOIN (AHashed (hash), AHashed (a), BHashedOnA (hash), BHashedOnA (aAndC), 'JoinComp_3', []) \n"\
      "\n"\
      "/* and extract the two atts and check for equality */ \n"\
      "AandBJoinedWithAExtracted (a, aAndC, aExtracted) <= APPLY (AandBJoined (a), AandBJoined (a, aAndC), 'JoinComp_3', 'self_0', []) \n"\
      "AandBJoinedWithBothExtracted (a, aAndC, aExtracted, otherA) <= APPLY (AandBJoinedWithAExtracted (aAndC), AandBJoinedWithAExtracted (a, aAndC, aExtracted), 'JoinComp_3', 'attAccess_1', []) \n"\
      "AandBJoinedWithBool (aAndC, a, bool) <= APPLY (AandBJoinedWithBothExtracted (aExtracted, otherA), AandBJoinedWithBothExtracted (aAndC, a), 'JoinComp_3', '==_2', []) \n"\
      "AandBJoinedFiltered (a, aAndC) <= FILTER (AandBJoinedWithBool (bool), AandBJoinedWithBool (a, aAndC), 'JoinComp_3', []) \n"\
      "\n"\
      "/* now get ready to join the strings */ \n"\
      "AandBJoinedFilteredWithC (a, aAndC, cExtracted) <= APPLY (AandBJoinedFiltered (aAndC), AandBJoinedFiltered (a, aAndC), 'JoinComp_3', 'attAccess_3', []) \n"\
      "BHashedOnC (a, aAndC, hash) <= HASHLEFT (AandBJoinedFilteredWithC (cExtracted), AandBJoinedFilteredWithC (a, aAndC), 'JoinComp_3', '==_5', []) \n"\
      "CwithCExtracted (c, cExtracted) <= APPLY (C (c), C (c), 'JoinComp_3', 'self_0', []) \n"\
      "CHashedOnC (c, hash) <= HASHRIGHT (CwithCExtracted (cExtracted), CwithCExtracted (c), 'JoinComp_3', '==_5', []) \n"\
      "\n"\
      "/* join the two of them */ \n"\
      "BandCJoined (a, aAndC, c) <= JOIN (BHashedOnC (hash), BHashedOnC (a, aAndC), CHashedOnC (hash), CHashedOnC (c), 'JoinComp_3', []) \n"\
      "\n"\
      "/* and extract the two atts and check for equality */ \n"\
      "BandCJoinedWithCExtracted (a, aAndC, c, cFromLeft) <= APPLY (BandCJoined (aAndC), BandCJoined (a, aAndC, c), 'JoinComp_3', 'attAccess_3', []) \n"\
      "BandCJoinedWithBoth (a, aAndC, c, cFromLeft, cFromRight) <= APPLY (BandCJoinedWithCExtracted (c), BandCJoinedWithCExtracted (a, aAndC, c, cFromLeft), 'JoinComp_3', 'self_4', []) \n"\
      "BandCJoinedWithBool (a, aAndC, c, bool) <= APPLY (BandCJoinedWithBoth (cFromLeft, cFromRight), BandCJoinedWithBoth (a, aAndC, c), 'JoinComp_3', '==_5', []) \n"\
      "last (a, aAndC, c) <= FILTER (BandCJoinedWithBool (bool), BandCJoinedWithBool (a, aAndC, c), 'JoinComp_3', []) \n"\
      "\n"\
      "/* and here is the answer */ \n"\
      "almostFinal (result) <= APPLY (last (a, aAndC, c), last (), 'JoinComp_3', 'native_lambda_7', []) \n"\
      "nothing () <= OUTPUT (almostFinal (result), 'outSet', 'myDB', 'SetWriter_4', [])";

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
  std::set<pdb::PDBAbstractPhysicalNodePtr> visitedNodes;

  // check size
  EXPECT_EQ(out.size(), 3);

  // chec pipelines
  while(!out.empty()) {

    auto firstComp = out.back()->getPipeComputations().front();

    if(firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "A") {

      int i = 0;
      for(auto &it : out.back()->getPipeComputations()) {
        switch (i) {

          case 0: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
            EXPECT_EQ(it->getOutputName(), "A");

            break;
          };
          case 1: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "AWithAExtracted");

            break;
          };
          case 2: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), HashLeftTypeID);
            EXPECT_EQ(it->getOutputName(), "AHashed");

            break;
          };
          default: { EXPECT_FALSE(true); break;};
        }

        i++;
      }

      // this must be the first time we visited this
      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
      visitedNodes.insert(out.back());

      // do we have one consumer
      EXPECT_EQ(out.back()->getConsumers().size(), 1);
      EXPECT_EQ(out.back()->getConsumers().front()->getPipeComputations().front()->getOutputName(), "AandBJoined");
    }
    else if(firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "B") {

      int i = 0;
      for(auto &it : out.back()->getPipeComputations()) {
        switch (i) {

          case 0: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
            EXPECT_EQ(it->getOutputName(), "B");

            break;
          };
          case 1: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "BWithAExtracted");

            break;
          };
          case 2: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), HashRightTypeID);
            EXPECT_EQ(it->getOutputName(), "BHashedOnA");

            break;
          };
          default: { EXPECT_FALSE(true); break;};
        }

        i++;
      }

      // this must be the first time we visited this
      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
      visitedNodes.insert(out.back());

      // do we have one consumer
      EXPECT_EQ(out.back()->getConsumers().size(), 1);
      EXPECT_EQ(out.back()->getConsumers().front()->getPipeComputations().front()->getOutputName(), "AandBJoined");
    }
    else if(firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "C") {

      int i = 0;
      for(auto &it : out.back()->getPipeComputations()) {
        switch (i) {

          case 0: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
            EXPECT_EQ(it->getOutputName(), "C");

            break;
          };
          case 1: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "CwithCExtracted");

            break;
          };
          case 2: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), HashRightTypeID);
            EXPECT_EQ(it->getOutputName(), "CHashedOnC");

            break;
          };
          default: { EXPECT_FALSE(true); break;};
        }

        i++;
      }

      // this must be the first time we visited this
      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
      visitedNodes.insert(out.back());

      // do we have one consumer
      EXPECT_EQ(out.back()->getConsumers().size(), 1);
      EXPECT_EQ(out.back()->getConsumers().front()->getPipeComputations().front()->getOutputName(), "BandCJoined");
    }
    else if(firstComp->getAtomicComputationTypeID() == ApplyJoinTypeID && firstComp->getOutputName() == "AandBJoined") {

      int i = 0;
      for(auto &it : out.back()->getPipeComputations()) {
        switch (i) {

          case 0: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyJoinTypeID);
            EXPECT_EQ(it->getOutputName(), "AandBJoined");

            break;
          };
          case 1: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "AandBJoinedWithAExtracted");

            break;
          };
          case 2: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "AandBJoinedWithBothExtracted");

            break;
          };
          case 3: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "AandBJoinedWithBool");

            break;
          };
          case 4: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
            EXPECT_EQ(it->getOutputName(), "AandBJoinedFiltered");

            break;
          };
          case 5: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "AandBJoinedFilteredWithC");

            break;
          };
          case 6: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), HashLeftTypeID);
            EXPECT_EQ(it->getOutputName(), "BHashedOnC");

            break;
          };
          default: { EXPECT_FALSE(true); break;};
        }

        i++;
      }

      // this must be the first time we visited this
      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
      visitedNodes.insert(out.back());

      // do we have one consumer
      EXPECT_EQ(out.back()->getConsumers().size(), 1);
      EXPECT_EQ(out.back()->getConsumers().front()->getPipeComputations().front()->getOutputName(), "BandCJoined");
    }
    else if(firstComp->getAtomicComputationTypeID() == ApplyJoinTypeID && firstComp->getOutputName() == "BandCJoined") {

      int i = 0;
      for(auto &it : out.back()->getPipeComputations()) {
        switch (i) {

          case 0: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyJoinTypeID);
            EXPECT_EQ(it->getOutputName(), "BandCJoined");

            break;
          };
          case 1: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "BandCJoinedWithCExtracted");

            break;
          };
          case 2: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "BandCJoinedWithBoth");

            break;
          };
          case 3: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "BandCJoinedWithBool");

            break;
          };
          case 4: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
            EXPECT_EQ(it->getOutputName(), "last");

            break;
          };
          case 5: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
            EXPECT_EQ(it->getOutputName(), "almostFinal");

            break;
          };
          case 6: {

            EXPECT_EQ(it->getAtomicComputationTypeID(), WriteSetTypeID);
            EXPECT_EQ(it->getOutputName(), "nothing");

            break;
          };
          default: {
            EXPECT_FALSE(true);
            break;
          };
        }

        ++i;
      }


      // this must be the first time we visited this
      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
      visitedNodes.insert(out.back());
    }
    else {
      std::cout << "sucks" << std::endl;
    }

    // get the last
    auto me = out.back();
    out.pop_back();

    // go through all consumers if they are not visited visit them
    for(auto &it : me->getConsumers()) {
      if(visitedNodes.find(it) == visitedNodes.end()) {
        out.push_back(it);
      }
    }
  }


}