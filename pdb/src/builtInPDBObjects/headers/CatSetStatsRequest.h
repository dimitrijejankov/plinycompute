#pragma once


// PRELOAD %CatSetStatsRequest%

namespace pdb {

// encapsulates a request to obtain a the stats of a set from a catalog
class CatSetStatsRequest : public Object {

public:

  CatSetStatsRequest() = default;
  CatSetStatsRequest(const std::string& dbNameIn, const std::string& setNameIn) : databaseName(dbNameIn), setName(setNameIn) {}
  ~CatSetStatsRequest() = default;

  ENABLE_DEEP_COPY

  String databaseName;
  String setName;
};

}
