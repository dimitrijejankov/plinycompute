#pragma once

#include <SetWriter.h>

namespace pdb {

class SillyWriteIntString : public SetWriter <String> {

public:

  SillyWriteIntString() = default;

  SillyWriteIntString(const String &db_name, const String &set_name) : SetWriter(db_name, set_name) {}

  ENABLE_DEEP_COPY

};

}
