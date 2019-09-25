#include <LogicalPlan.h>
#include <Lexer.h>
#include <Parser.h>

namespace pdb {

LogicalPlan::LogicalPlan(const std::string &tcap, Vector<Handle<Computation>> &computations)  {
  init(tcap, computations);
}

LogicalPlan::LogicalPlan(AtomicComputationList &computationsIn, pdb::Vector<pdb::Handle<pdb::Computation>> &allComputations) {
  init(computationsIn, allComputations);
}

void LogicalPlan::init(AtomicComputationList &computationsIn, pdb::Vector<pdb::Handle<pdb::Computation>> &allComputations) {
  computations = computationsIn;
  for (int i = 0; i < allComputations.size(); i++) {
    std::string compType = allComputations[i]->getComputationType();
    compType += "_";
    compType += std::to_string(i);
    pdb::ComputationNode temp(allComputations[i]);
    allConstituentComputations[compType] = temp;
  }
}

void LogicalPlan::init(const std::string &tcap, Vector<Handle<Computation>> &allComputations) {

  // get the string to compile
  std::string myLogicalPlan = tcap;
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

  // copy all the computations
  init(*myResult, allComputations);

  delete myResult;
}

}
