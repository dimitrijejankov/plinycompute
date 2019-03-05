//
// Created by dimitrije on 2/21/19.
//

#include <physicalOptimizer/PDBPhysicalOptimizer.h>
#include <AtomicComputationList.h>
#include <Lexer.h>
#include <Parser.h>
#include <SetScanner.h>
#include <AtomicComputationClasses.h>
#include <PDBCatalogClient.h>

namespace pdb {

PDBPhysicalOptimizer::PDBPhysicalOptimizer(String tcapString,
                                           PDBCatalogClientPtr &clientPtr,
                                           PDBLoggerPtr &logger) : logger(logger), sources(([](const OptimizerSource& s1, const OptimizerSource& s2) {return s1.first > s2.first;})){

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
  pdb::PDBPipeNodeBuilder factory(computationID, atomicComputations);

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
    auto set = clientPtr->getSet(setIdentifier.first, setIdentifier.second, error);

    if(set == nullptr) {
      throw runtime_error("Could not find the set I needed. " +  error);
    }

    // add the source to the pq
    sources.push(std::make_pair(set->setSize, source));
  }
}

pdb::Handle<pdb::PDBPhysicalAlgorithm> PDBPhysicalOptimizer::getNextAlgorithm() {

  // select a source
  auto source = sources.top();
  sources.pop();

  // runs the algorithm generation part
  return source.second->generateAlgorithm();
}

bool PDBPhysicalOptimizer::hasAlgorithmToRun() {
  return !sources.empty();
}

void PDBPhysicalOptimizer::updateStats() {

}

}

