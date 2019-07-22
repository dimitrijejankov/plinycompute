//
// Created by dimitrije on 4/9/19.
//

#ifndef PDB_SILLYJOIN_H
#define PDB_SILLYJOIN_H

#include <JoinComp.h>
#include <LambdaCreationFunctions.h>

namespace pdb {

// this plan has three tables: A (a: int), B (a: int, c: String), C (c: int)
// it first joins A with B, and then joins the result with C
class SillyJoin : public JoinComp <SillyJoin, String, int, StringIntPair, String> {

public:

  ENABLE_DEEP_COPY

  Lambda <bool> getSelection (Handle <int> in1, Handle <StringIntPair> in2, Handle <String> in3) {
    return (makeLambdaFromSelf (in1) == makeLambdaFromMember (in2, myInt)) && (makeLambdaFromMember (in2, myString) == makeLambdaFromSelf (in3));
  }

  Lambda <Handle <String>> getProjection (Handle <int> in1, Handle <StringIntPair> in2, Handle <String> in3) {
    return makeLambda (in1, in2, in3, [] (Handle <int> &in1, Handle <StringIntPair> &in2, Handle <String> &in3) {
      std::ostringstream oss;
      oss << "Got int " << *in1 << " and StringIntPair (" << in2->myInt << ", '" << *(in2->myString) << "') and String '" << *in3 << "'";
      Handle <String> res = makeObject <String> (oss.str ());
      return res;
    });
  }
};

}

#endif //PDB_SILLYJOIN_H
