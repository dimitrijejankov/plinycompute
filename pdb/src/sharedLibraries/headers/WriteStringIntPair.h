#pragma once

#include <SetWriter.h>
#include <StringIntPair.h>

namespace pdb {

class WriteStringIntPair : public SetWriter <StringIntPair> {
public:

  WriteStringIntPair() = default;
  WriteStringIntPair(const String &db_name, const String &set_name) : SetWriter(db_name, set_name) {}

  ENABLE_DEEP_COPY


};

}

