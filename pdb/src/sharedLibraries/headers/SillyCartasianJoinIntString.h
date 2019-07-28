#pragma once

#include <JoinComp.h>
#include <LambdaCreationFunctions.h>

namespace pdb {

class SillyCartasianJoinIntString : public JoinComp <SillyCartasianJoinIntString, String, int, StringIntPair> {
public:

  ENABLE_DEEP_COPY

  static Lambda <bool> getSelection (Handle <int> in1, Handle <StringIntPair> in2) {
    return makeLambda (in1, [] (Handle <int> &in1) { return true;}) && (makeLambdaFromMethod (in2, getBoolean));
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