//
// Created by dimitrije on 2/22/19.
//

#include <gtest/gtest.h>
#include <AtomicComputationList.h>
#include <Parser.h>
#include <PDBPipeNodeBuilder.h>
#include <PDBJoinPhysicalNode.h>

TEST(TestPipeBuilder, Test1) {


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

  pdb::PDBPipeNodeBuilder factory(1, atomicComputations);

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

TEST(TestPipeBuilder, Test2) {

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

  pdb::PDBPipeNodeBuilder factory(2, atomicComputations);

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

TEST(TestPipeBuilder, Test3) {
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

  pdb::PDBPipeNodeBuilder factory(3, atomicComputations);

  auto out = factory.generateAnalyzerGraph();
  std::set<pdb::PDBAbstractPhysicalNodePtr> visitedNodes;

  // check size
  EXPECT_EQ(out.size(), 3);

  // chec preaggregationPipelines
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

      // check the other side
      auto otherSide = ((pdb::PDBJoinPhysicalNode*) out.back().get())->otherSide.lock();
      firstComp = otherSide->getPipeComputations().front();
      EXPECT_TRUE(firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "B");

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

      // check the other side
      auto otherSide = ((pdb::PDBJoinPhysicalNode*) out.back().get())->otherSide.lock();
      firstComp = otherSide->getPipeComputations().front();
      EXPECT_TRUE(firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "A");

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

      // check the other side
      auto otherSide = ((pdb::PDBJoinPhysicalNode*) out.back().get())->otherSide.lock();
      firstComp = otherSide->getPipeComputations().front();
      EXPECT_TRUE(firstComp->getAtomicComputationTypeID() == ApplyJoinTypeID && firstComp->getOutputName() == "AandBJoined");

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

      // check the other side
      auto otherSide = ((pdb::PDBJoinPhysicalNode*) out.back().get())->otherSide.lock();
      firstComp = otherSide->getPipeComputations().front();
      EXPECT_TRUE(firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "C");
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
      EXPECT_FALSE(true);
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

TEST(TestPipeBuilder, Test4) {
  std::string myLogicalPlan = "inputDataForSetScanner_0(in0) <= SCAN ('LDA_db', 'LDA_input_set', 'SetScanner_0')\n"
                              "\n"
                              "/* Extract key for aggregation */\n"
                              "nativ_0OutForAggregationComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'AggregationComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                              "\n"
                              "/* Extract value for aggregation */\n"
                              "nativ_1OutForAggregationComp1(nativ_0_1OutFor,nativ_1_1OutFor) <= APPLY (nativ_0OutForAggregationComp1(in0), nativ_0OutForAggregationComp1(nativ_0_1OutFor), 'AggregationComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                              "\n"
                              "/* Apply aggregation */\n"
                              "aggOutForAggregationComp1 (aggOutFor1)<= AGGREGATE (nativ_1OutForAggregationComp1(nativ_0_1OutFor, nativ_1_1OutFor),'AggregationComp_1')\n"
                              "\n"
                              "/* Apply selection filtering */\n"
                              "nativ_0OutForSelectionComp2(aggOutFor1,nativ_0_2OutFor) <= APPLY (aggOutForAggregationComp1(aggOutFor1), aggOutForAggregationComp1(aggOutFor1), 'SelectionComp_2', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                              "filteredInputForSelectionComp2(aggOutFor1) <= FILTER (nativ_0OutForSelectionComp2(nativ_0_2OutFor), nativ_0OutForSelectionComp2(aggOutFor1), 'SelectionComp_2')\n"
                              "\n"
                              "/* Apply selection projection */\n"
                              "nativ_1OutForSelectionComp2 (nativ_1_2OutFor) <= APPLY (filteredInputForSelectionComp2(aggOutFor1), filteredInputForSelectionComp2(), 'SelectionComp_2', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                              "inputDataForSetScanner_3(in3) <= SCAN ('LDA_db', 'LDA_meta_data_set', 'SetScanner_3')\n"
                              "\n"
                              "/* Apply selection filtering */\n"
                              "nativ_0OutForSelectionComp4(in3,nativ_0_4OutFor) <= APPLY (inputDataForSetScanner_3(in3), inputDataForSetScanner_3(in3), 'SelectionComp_4', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                              "filteredInputForSelectionComp4(in3) <= FILTER (nativ_0OutForSelectionComp4(nativ_0_4OutFor), nativ_0OutForSelectionComp4(in3), 'SelectionComp_4')\n"
                              "\n"
                              "/* Apply selection projection */\n"
                              "nativ_1OutForSelectionComp4 (nativ_1_4OutFor) <= APPLY (filteredInputForSelectionComp4(in3), filteredInputForSelectionComp4(), 'SelectionComp_4', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                              "attAccess_0ExtractedForJoinComp5(in0,att_0ExtractedFor_docID) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'JoinComp_5', 'attAccess_0', [('attName', 'docID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDADocument'), ('lambdaType', 'attAccess')])\n"
                              "attAccess_0ExtractedForJoinComp5_hashed(in0,att_0ExtractedFor_docID_hash) <= HASHLEFT (attAccess_0ExtractedForJoinComp5(att_0ExtractedFor_docID), attAccess_0ExtractedForJoinComp5(in0), 'JoinComp_5', '==_2', [])\n"
                              "attAccess_1ExtractedForJoinComp5(nativ_1_2OutFor,att_1ExtractedFor_myInt) <= APPLY (nativ_1OutForSelectionComp2(nativ_1_2OutFor), nativ_1OutForSelectionComp2(nativ_1_2OutFor), 'JoinComp_5', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'IntDoubleVectorPair'), ('lambdaType', 'attAccess')])\n"
                              "attAccess_1ExtractedForJoinComp5_hashed(nativ_1_2OutFor,att_1ExtractedFor_myInt_hash) <= HASHRIGHT (attAccess_1ExtractedForJoinComp5(att_1ExtractedFor_myInt), attAccess_1ExtractedForJoinComp5(nativ_1_2OutFor), 'JoinComp_5', '==_2', [])\n"
                              "\n"
                              "/* Join ( in0 ) and ( nativ_1_2OutFor ) */\n"
                              "JoinedFor_equals2JoinComp5(in0, nativ_1_2OutFor) <= JOIN (attAccess_0ExtractedForJoinComp5_hashed(att_0ExtractedFor_docID_hash), attAccess_0ExtractedForJoinComp5_hashed(in0), attAccess_1ExtractedForJoinComp5_hashed(att_1ExtractedFor_myInt_hash), attAccess_1ExtractedForJoinComp5_hashed(nativ_1_2OutFor), 'JoinComp_5')\n"
                              "JoinedFor_equals2JoinComp5_WithLHSExtracted(in0,nativ_1_2OutFor,LHSExtractedFor_2_5) <= APPLY (JoinedFor_equals2JoinComp5(in0), JoinedFor_equals2JoinComp5(in0,nativ_1_2OutFor), 'JoinComp_5', 'attAccess_0', [('attName', 'docID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDADocument'), ('lambdaType', 'attAccess')])\n"
                              "JoinedFor_equals2JoinComp5_WithBOTHExtracted(in0,nativ_1_2OutFor,LHSExtractedFor_2_5,RHSExtractedFor_2_5) <= APPLY (JoinedFor_equals2JoinComp5_WithLHSExtracted(nativ_1_2OutFor), JoinedFor_equals2JoinComp5_WithLHSExtracted(in0,nativ_1_2OutFor,LHSExtractedFor_2_5), 'JoinComp_5', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'IntDoubleVectorPair'), ('lambdaType', 'attAccess')])\n"
                              "JoinedFor_equals2JoinComp5_BOOL(in0,nativ_1_2OutFor,bool_2_5) <= APPLY (JoinedFor_equals2JoinComp5_WithBOTHExtracted(LHSExtractedFor_2_5,RHSExtractedFor_2_5), JoinedFor_equals2JoinComp5_WithBOTHExtracted(in0,nativ_1_2OutFor), 'JoinComp_5', '==_2', [('lambdaType', '==')])\n"
                              "JoinedFor_equals2JoinComp5_FILTERED(in0, nativ_1_2OutFor) <= FILTER (JoinedFor_equals2JoinComp5_BOOL(bool_2_5), JoinedFor_equals2JoinComp5_BOOL(in0, nativ_1_2OutFor), 'JoinComp_5')\n"
                              "attAccess_3ExtractedForJoinComp5(in0,nativ_1_2OutFor,att_3ExtractedFor_wordID) <= APPLY (JoinedFor_equals2JoinComp5_FILTERED(in0), JoinedFor_equals2JoinComp5_FILTERED(in0,nativ_1_2OutFor), 'JoinComp_5', 'attAccess_3', [('attName', 'wordID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDADocument'), ('lambdaType', 'attAccess')])\n"
                              "attAccess_3ExtractedForJoinComp5_hashed(in0,nativ_1_2OutFor,att_3ExtractedFor_wordID_hash) <= HASHLEFT (attAccess_3ExtractedForJoinComp5(att_3ExtractedFor_wordID), attAccess_3ExtractedForJoinComp5(in0,nativ_1_2OutFor), 'JoinComp_5', '==_5', [])\n"
                              "attAccess_4ExtractedForJoinComp5(nativ_1_4OutFor,att_4ExtractedFor_whichWord) <= APPLY (nativ_1OutForSelectionComp4(nativ_1_4OutFor), nativ_1OutForSelectionComp4(nativ_1_4OutFor), 'JoinComp_5', 'attAccess_4', [('attName', 'whichWord'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDATopicWordProb'), ('lambdaType', 'attAccess')])\n"
                              "attAccess_4ExtractedForJoinComp5_hashed(nativ_1_4OutFor,att_4ExtractedFor_whichWord_hash) <= HASHRIGHT (attAccess_4ExtractedForJoinComp5(att_4ExtractedFor_whichWord), attAccess_4ExtractedForJoinComp5(nativ_1_4OutFor), 'JoinComp_5', '==_5', [])\n"
                              "\n"
                              "/* Join ( in0 nativ_1_2OutFor ) and ( nativ_1_4OutFor ) */\n"
                              "JoinedFor_equals5JoinComp5(in0, nativ_1_2OutFor, nativ_1_4OutFor) <= JOIN (attAccess_3ExtractedForJoinComp5_hashed(att_3ExtractedFor_wordID_hash), attAccess_3ExtractedForJoinComp5_hashed(in0, nativ_1_2OutFor), attAccess_4ExtractedForJoinComp5_hashed(att_4ExtractedFor_whichWord_hash), attAccess_4ExtractedForJoinComp5_hashed(nativ_1_4OutFor), 'JoinComp_5')\n"
                              "JoinedFor_equals5JoinComp5_WithLHSExtracted(in0,nativ_1_2OutFor,nativ_1_4OutFor,LHSExtractedFor_5_5) <= APPLY (JoinedFor_equals5JoinComp5(in0), JoinedFor_equals5JoinComp5(in0,nativ_1_2OutFor,nativ_1_4OutFor), 'JoinComp_5', 'attAccess_3', [('attName', 'wordID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDADocument'), ('lambdaType', 'attAccess')])\n"
                              "JoinedFor_equals5JoinComp5_WithBOTHExtracted(in0,nativ_1_2OutFor,nativ_1_4OutFor,LHSExtractedFor_5_5,RHSExtractedFor_5_5) <= APPLY (JoinedFor_equals5JoinComp5_WithLHSExtracted(nativ_1_4OutFor), JoinedFor_equals5JoinComp5_WithLHSExtracted(in0,nativ_1_2OutFor,nativ_1_4OutFor,LHSExtractedFor_5_5), 'JoinComp_5', 'attAccess_4', [('attName', 'whichWord'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDATopicWordProb'), ('lambdaType', 'attAccess')])\n"
                              "JoinedFor_equals5JoinComp5_BOOL(in0,nativ_1_2OutFor,nativ_1_4OutFor,bool_5_5) <= APPLY (JoinedFor_equals5JoinComp5_WithBOTHExtracted(LHSExtractedFor_5_5,RHSExtractedFor_5_5), JoinedFor_equals5JoinComp5_WithBOTHExtracted(in0,nativ_1_2OutFor,nativ_1_4OutFor), 'JoinComp_5', '==_5', [('lambdaType', '==')])\n"
                              "JoinedFor_equals5JoinComp5_FILTERED(in0, nativ_1_2OutFor, nativ_1_4OutFor) <= FILTER (JoinedFor_equals5JoinComp5_BOOL(bool_5_5), JoinedFor_equals5JoinComp5_BOOL(in0, nativ_1_2OutFor, nativ_1_4OutFor), 'JoinComp_5')\n"
                              "\n"
                              "/* run Join projection on ( in0 nativ_1_2OutFor nativ_1_4OutFor )*/\n"
                              "nativ_7OutForJoinComp5 (nativ_7_5OutFor) <= APPLY (JoinedFor_equals5JoinComp5_FILTERED(in0,nativ_1_2OutFor,nativ_1_4OutFor), JoinedFor_equals5JoinComp5_FILTERED(), 'JoinComp_5', 'native_lambda_7', [('lambdaType', 'native_lambda')])\n"
                              "\n"
                              "/* Apply selection filtering */\n"
                              "nativ_0OutForSelectionComp6(nativ_7_5OutFor,nativ_0_6OutFor) <= APPLY (nativ_7OutForJoinComp5(nativ_7_5OutFor), nativ_7OutForJoinComp5(nativ_7_5OutFor), 'SelectionComp_6', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                              "filteredInputForSelectionComp6(nativ_7_5OutFor) <= FILTER (nativ_0OutForSelectionComp6(nativ_0_6OutFor), nativ_0OutForSelectionComp6(nativ_7_5OutFor), 'SelectionComp_6')\n"
                              "\n"
                              "/* Apply selection projection */\n"
                              "nativ_1OutForSelectionComp6 (nativ_1_6OutFor) <= APPLY (filteredInputForSelectionComp6(nativ_7_5OutFor), filteredInputForSelectionComp6(), 'SelectionComp_6', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                              "\n"
                              "/* Apply MultiSelection filtering */\n"
                              "nativ_0OutForMultiSelectionComp7(nativ_1_6OutFor,nativ_0_7OutFor) <= APPLY (nativ_1OutForSelectionComp6(nativ_1_6OutFor), nativ_1OutForSelectionComp6(nativ_1_6OutFor), 'MultiSelectionComp_7', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                              "filteredInputForMultiSelectionComp7(nativ_1_6OutFor) <= FILTER (nativ_0OutForMultiSelectionComp7(nativ_0_7OutFor), nativ_0OutForMultiSelectionComp7(nativ_1_6OutFor), 'MultiSelectionComp_7')\n"
                              "\n"
                              "/* Apply MultiSelection projection */\n"
                              "methodCall_1OutFor_MultiSelectionComp7(nativ_1_6OutFor,methodCall_1OutFor__getTopicAssigns) <= APPLY (filteredInputForMultiSelectionComp7(nativ_1_6OutFor), filteredInputForMultiSelectionComp7(nativ_1_6OutFor), 'MultiSelectionComp_7', 'methodCall_1', [('inputTypeName', 'LDADocWordTopicAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getTopicAssigns'), ('returnTypeName', 'LDADocWordTopicAssignment')])\n"
                              "deref_2OutForMultiSelectionComp7 (methodCall_1OutFor__getTopicAssigns) <= APPLY (methodCall_1OutFor_MultiSelectionComp7(methodCall_1OutFor__getTopicAssigns), methodCall_1OutFor_MultiSelectionComp7(), 'MultiSelectionComp_7', 'deref_2')\n"
                              "flattenedOutForMultiSelectionComp7(flattened_methodCall_1OutFor__getTopicAssigns) <= FLATTEN (deref_2OutForMultiSelectionComp7(methodCall_1OutFor__getTopicAssigns), deref_2OutForMultiSelectionComp7(), 'MultiSelectionComp_7')\n"
                              "\n"
                              "/* Extract key for aggregation */\n"
                              "methodCall_0OutFor_AggregationComp8(flattened_methodCall_1OutFor__getTopicAssigns,methodCall_0OutFor__getKey) <= APPLY (flattenedOutForMultiSelectionComp7(flattened_methodCall_1OutFor__getTopicAssigns), flattenedOutForMultiSelectionComp7(flattened_methodCall_1OutFor__getTopicAssigns), 'AggregationComp_8', 'methodCall_0', [('inputTypeName', 'TopicAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getKey'), ('returnTypeName', 'TopicAssignment')])\n"
                              "deref_1OutForAggregationComp8(flattened_methodCall_1OutFor__getTopicAssigns, methodCall_0OutFor__getKey) <= APPLY (methodCall_0OutFor_AggregationComp8(methodCall_0OutFor__getKey), methodCall_0OutFor_AggregationComp8(flattened_methodCall_1OutFor__getTopicAssigns), 'AggregationComp_8', 'deref_1')\n"
                              "\n"
                              "/* Extract value for aggregation */\n"
                              "methodCall_2OutFor_AggregationComp8(methodCall_0OutFor__getKey,methodCall_2OutFor__getValue) <= APPLY (deref_1OutForAggregationComp8(flattened_methodCall_1OutFor__getTopicAssigns), deref_1OutForAggregationComp8(methodCall_0OutFor__getKey), 'AggregationComp_8', 'methodCall_2', [('inputTypeName', 'TopicAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getValue'), ('returnTypeName', 'TopicAssignment')])\n"
                              "deref_3OutForAggregationComp8(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue) <= APPLY (methodCall_2OutFor_AggregationComp8(methodCall_2OutFor__getValue), methodCall_2OutFor_AggregationComp8(methodCall_0OutFor__getKey), 'AggregationComp_8', 'deref_3')\n"
                              "\n"
                              "/* Apply aggregation */\n"
                              "aggOutForAggregationComp8 (aggOutFor8)<= AGGREGATE (deref_3OutForAggregationComp8(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue),'AggregationComp_8')\n"
                              "\n"
                              "/* Apply MultiSelection filtering */\n"
                              "nativ_0OutForMultiSelectionComp9(aggOutFor8,nativ_0_9OutFor) <= APPLY (aggOutForAggregationComp8(aggOutFor8), aggOutForAggregationComp8(aggOutFor8), 'MultiSelectionComp_9', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                              "filteredInputForMultiSelectionComp9(aggOutFor8) <= FILTER (nativ_0OutForMultiSelectionComp9(nativ_0_9OutFor), nativ_0OutForMultiSelectionComp9(aggOutFor8), 'MultiSelectionComp_9')\n"
                              "\n"
                              "/* Apply MultiSelection projection */\n"
                              "nativ_1OutForMultiSelectionComp9 (nativ_1_9OutFor) <= APPLY (filteredInputForMultiSelectionComp9(aggOutFor8), filteredInputForMultiSelectionComp9(), 'MultiSelectionComp_9', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                              "flattenedOutForMultiSelectionComp9(flattened_nativ_1_9OutFor) <= FLATTEN (nativ_1OutForMultiSelectionComp9(nativ_1_9OutFor), nativ_1OutForMultiSelectionComp9(), 'MultiSelectionComp_9')\n"
                              "\n"
                              "/* Extract key for aggregation */\n"
                              "methodCall_0OutFor_AggregationComp10(flattened_nativ_1_9OutFor,methodCall_0OutFor__getKey) <= APPLY (flattenedOutForMultiSelectionComp9(flattened_nativ_1_9OutFor), flattenedOutForMultiSelectionComp9(flattened_nativ_1_9OutFor), 'AggregationComp_10', 'methodCall_0', [('inputTypeName', 'LDATopicWordProb'), ('lambdaType', 'methodCall'), ('methodName', 'getKey'), ('returnTypeName', 'LDATopicWordProb')])\n"
                              "deref_1OutForAggregationComp10(flattened_nativ_1_9OutFor, methodCall_0OutFor__getKey) <= APPLY (methodCall_0OutFor_AggregationComp10(methodCall_0OutFor__getKey), methodCall_0OutFor_AggregationComp10(flattened_nativ_1_9OutFor), 'AggregationComp_10', 'deref_1')\n"
                              "\n"
                              "/* Extract value for aggregation */\n"
                              "methodCall_2OutFor_AggregationComp10(methodCall_0OutFor__getKey,methodCall_2OutFor__getValue) <= APPLY (deref_1OutForAggregationComp10(flattened_nativ_1_9OutFor), deref_1OutForAggregationComp10(methodCall_0OutFor__getKey), 'AggregationComp_10', 'methodCall_2', [('inputTypeName', 'LDATopicWordProb'), ('lambdaType', 'methodCall'), ('methodName', 'getValue'), ('returnTypeName', 'LDATopicWordProb')])\n"
                              "deref_3OutForAggregationComp10(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue) <= APPLY (methodCall_2OutFor_AggregationComp10(methodCall_2OutFor__getValue), methodCall_2OutFor_AggregationComp10(methodCall_0OutFor__getKey), 'AggregationComp_10', 'deref_3')\n"
                              "\n"
                              "/* Apply aggregation */\n"
                              "aggOutForAggregationComp10 (aggOutFor10)<= AGGREGATE (deref_3OutForAggregationComp10(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue),'AggregationComp_10')\n"
                              "aggOutForAggregationComp10_out( ) <= OUTPUT ( aggOutForAggregationComp10 ( aggOutFor10 ), 'LDA_db', 'TopicsPerWord1', 'SetWriter_11')\n"
                              "\n"
                              "/* Apply MultiSelection filtering */\n"
                              "nativ_0OutForMultiSelectionComp12(nativ_1_6OutFor,nativ_0_12OutFor) <= APPLY (nativ_1OutForSelectionComp6(nativ_1_6OutFor), nativ_1OutForSelectionComp6(nativ_1_6OutFor), 'MultiSelectionComp_12', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                              "filteredInputForMultiSelectionComp12(nativ_1_6OutFor) <= FILTER (nativ_0OutForMultiSelectionComp12(nativ_0_12OutFor), nativ_0OutForMultiSelectionComp12(nativ_1_6OutFor), 'MultiSelectionComp_12')\n"
                              "\n"
                              "/* Apply MultiSelection projection */\n"
                              "methodCall_1OutFor_MultiSelectionComp12(nativ_1_6OutFor,methodCall_1OutFor__getDocAssigns) <= APPLY (filteredInputForMultiSelectionComp12(nativ_1_6OutFor), filteredInputForMultiSelectionComp12(nativ_1_6OutFor), 'MultiSelectionComp_12', 'methodCall_1', [('inputTypeName', 'LDADocWordTopicAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getDocAssigns'), ('returnTypeName', 'LDADocWordTopicAssignment')])\n"
                              "deref_2OutForMultiSelectionComp12 (methodCall_1OutFor__getDocAssigns) <= APPLY (methodCall_1OutFor_MultiSelectionComp12(methodCall_1OutFor__getDocAssigns), methodCall_1OutFor_MultiSelectionComp12(), 'MultiSelectionComp_12', 'deref_2')\n"
                              "flattenedOutForMultiSelectionComp12(flattened_methodCall_1OutFor__getDocAssigns) <= FLATTEN (deref_2OutForMultiSelectionComp12(methodCall_1OutFor__getDocAssigns), deref_2OutForMultiSelectionComp12(), 'MultiSelectionComp_12')\n"
                              "\n"
                              "/* Extract key for aggregation */\n"
                              "methodCall_0OutFor_AggregationComp13(flattened_methodCall_1OutFor__getDocAssigns,methodCall_0OutFor__getKey) <= APPLY (flattenedOutForMultiSelectionComp12(flattened_methodCall_1OutFor__getDocAssigns), flattenedOutForMultiSelectionComp12(flattened_methodCall_1OutFor__getDocAssigns), 'AggregationComp_13', 'methodCall_0', [('inputTypeName', 'DocAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getKey'), ('returnTypeName', 'DocAssignment')])\n"
                              "deref_1OutForAggregationComp13(flattened_methodCall_1OutFor__getDocAssigns, methodCall_0OutFor__getKey) <= APPLY (methodCall_0OutFor_AggregationComp13(methodCall_0OutFor__getKey), methodCall_0OutFor_AggregationComp13(flattened_methodCall_1OutFor__getDocAssigns), 'AggregationComp_13', 'deref_1')\n"
                              "\n"
                              "/* Extract value for aggregation */\n"
                              "methodCall_2OutFor_AggregationComp13(methodCall_0OutFor__getKey,methodCall_2OutFor__getValue) <= APPLY (deref_1OutForAggregationComp13(flattened_methodCall_1OutFor__getDocAssigns), deref_1OutForAggregationComp13(methodCall_0OutFor__getKey), 'AggregationComp_13', 'methodCall_2', [('inputTypeName', 'DocAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getValue'), ('returnTypeName', 'DocAssignment')])\n"
                              "deref_3OutForAggregationComp13(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue) <= APPLY (methodCall_2OutFor_AggregationComp13(methodCall_2OutFor__getValue), methodCall_2OutFor_AggregationComp13(methodCall_0OutFor__getKey), 'AggregationComp_13', 'deref_3')\n"
                              "\n"
                              "/* Apply aggregation */\n"
                              "aggOutForAggregationComp13 (aggOutFor13)<= AGGREGATE (deref_3OutForAggregationComp13(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue),'AggregationComp_13')\n"
                              "\n"
                              "/* Apply selection filtering */\n"
                              "nativ_0OutForSelectionComp14(aggOutFor13,nativ_0_14OutFor) <= APPLY (aggOutForAggregationComp13(aggOutFor13), aggOutForAggregationComp13(aggOutFor13), 'SelectionComp_14', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                              "filteredInputForSelectionComp14(aggOutFor13) <= FILTER (nativ_0OutForSelectionComp14(nativ_0_14OutFor), nativ_0OutForSelectionComp14(aggOutFor13), 'SelectionComp_14')\n"
                              "\n"
                              "/* Apply selection projection */\n"
                              "nativ_1OutForSelectionComp14 (nativ_1_14OutFor) <= APPLY (filteredInputForSelectionComp14(aggOutFor13), filteredInputForSelectionComp14(), 'SelectionComp_14', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                              "nativ_1OutForSelectionComp14_out( ) <= OUTPUT ( nativ_1OutForSelectionComp14 ( nativ_1_14OutFor ), 'LDA_db', 'TopicsPerDoc1', 'SetWriter_15')";

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

  pdb::PDBPipeNodeBuilder factory(3, atomicComputations);

  set<pdb::PDBAbstractPhysicalNodePtr> check;
  auto out = factory.generateAnalyzerGraph();

  for(int j = 0; j < out.size(); ++j) {

    std::cout << "Pipeline " << j << std::endl;

    auto &o = out[j];

    for(auto &c : o->getPipeComputations()) {
      std::cout << c->getOutputName() << std::endl;
    }

    std::cout << "Num consumers " << o->getConsumers().size() << std::endl;

    for(auto &con : o->getConsumers()) {

      if(check.find(con) == check.end()) {
        out.push_back(con);
        check.insert(con);
      }

      std::cout << "Starts  : " << con->getPipeComputations().front()->getOutputName() << std::endl;
    }

    std::cout << "\n\n\n";
  }

}

TEST(TestPipeBuilder, Test5) {

  std::string myLogicalPlan = "inputDataForSetScanner_0(in0) <= SCAN ('chris_db', 'input_set1', 'SetScanner_0')\n"
                              "inputDataForSetScanner_1(in1) <= SCAN ('chris_db', 'input_set2', 'SetScanner_1')\n"
                              "unionOutUnionComp2 (unionOutFor2 )<= UNION (inputDataForSetScanner_0(in0), inputDataForSetScanner_1(in1),'UnionComp_2')\n"
                              "unionOutUnionComp2_out( ) <= OUTPUT ( unionOutUnionComp2 ( unionOutFor2 ), 'chris_db', 'outputSet', 'SetWriter_3')";

  // get the string to compile
  myLogicalPlan.push_back('\0');

  // where the result of the parse goes
  AtomicComputationList *myResult;

  // now, do the compilation
  yyscan_t scanner;
  LexerExtra extra{""};
  yylex_init_extra(&extra, &scanner);
  const YY_BUFFER_STATE buffer{yy_scan_string(myLogicalPlan.data(), scanner)};
  const int parseFailed = yyparse(scanner, &myResult);
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  // if it didn't parse, get outta here
  if (parseFailed) {
    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
    exit(1);
  }

  // this is the logical plan to return
  auto atomicComputations = std::shared_ptr<AtomicComputationList>(myResult);

  pdb::PDBPipeNodeBuilder factory(3, atomicComputations);

  set<pdb::PDBAbstractPhysicalNodePtr> check;
  auto out = factory.generateAnalyzerGraph();

  for(int j = 0; j < out.size(); ++j) {

    std::cout << "\nPipeline " << j << std::endl;

    auto &o = out[j];

    for(auto &c : o->getPipeComputations()) {
      std::cout << c->getOutputName() << std::endl;
    }

    std::cout << "Num consumers " << o->getConsumers().size() << std::endl;

    for(auto &con : o->getConsumers()) {

      if(check.find(con) == check.end()) {
        out.push_back(con);
        check.insert(con);
      }

      std::cout << "Starts  : " << con->getPipeComputations().front()->getOutputName() << std::endl;
    }

    std::cout << "\n\n\n";
  }

}
