//
// Created by dimitrije on 3/19/19.
//

#ifndef PDB_PDBPHYSICALOPTIMIZERTEMPLATE_H
#define PDB_PDBPHYSICALOPTIMIZERTEMPLATE_H

#include <Lexer.h>
#include <Parser.h>
#include <PDBCatalogClient.h>

namespace pdb {

template<class CatalogClient>
PDBPhysicalOptimizer::PDBPhysicalOptimizer(uint64_t computationID,
                                           const String& tcapString,
                                           const std::unordered_map<uint64_t, bool>& keyedComputations,
                                           const shared_ptr<CatalogClient> &clientPtr,
                                           PDBLoggerPtr &logger) {

  // get the string to compile
  std::string myLogicalPlan = tcapString;
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

  // split the computations into pipes
  pdb::PDBPipeNodeBuilder factory(computationID,
                                  keyedComputations,
                                  atomicComputations);

  // fill the sources up
  auto sourcesVector = factory.generateAnalyzerGraph();
  for(const auto &source : sourcesVector) {

    // if the source does not have a scan set something went horribly wrong
    if(!source->hasScanSet()) {
      throw runtime_error("There was a pipe at the beginning of the query plan without a scan set");
    }

    // get the set
    std::string error;
    auto setIdentifier = source->getSourceSet();
    auto stats = clientPtr->getSetStats(setIdentifier.first, setIdentifier.second, error);

    if(stats == nullptr) {
      throw runtime_error("Could not find the set I needed. " +  error);
    }

    // add the source to the data structures
    sources.insert(std::make_pair(stats, source));
    pageSetCosts[source->getSourcePageSet(pageSetCosts)->pageSetIdentifier] = stats;
  }

}

}

#endif //PDB_PDBPHYSICALOPTIMIZERTEMPLATE_H
