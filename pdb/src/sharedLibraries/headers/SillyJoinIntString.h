#pragma once

#include <JoinComp.h>
#include <LambdaCreationFunctions.h>

namespace pdb {

class SillyJoinIntString : public JoinComp <SillyJoinIntString, String, int, StringIntPair> {
public:

  ENABLE_DEEP_COPY

  static Lambda <bool> getSelection (Handle <int> in1, Handle <StringIntPair> in2) {
    return (makeLambdaFromSelf (in1) == makeLambdaFromMember (in2, myInt));
  }

  static Lambda <Handle <String>> getProjection (Handle <int> in1, Handle <StringIntPair> in2) {
    return makeLambda (in1, in2, [] (Handle <int> &in1, Handle <StringIntPair> &in2) {
      std::ostringstream oss;
      oss << "Got int " << *in1 << " and StringIntPair (" << in2->myInt << ", '" << *(in2->myString) << "')'";
      Handle <String> res = makeObject <String> (oss.str ());
      return res;
    });
  }
};

}