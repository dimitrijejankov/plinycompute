/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#pragma once

#include "TupleSpec.h"
#include "AtomicComputationList.h"

#include "KeyValueList.h"

// NOTE: these are NOT part of the pdb namespace because they need to be included in an "extern
// C"...
// I am not sure whether that is possible... perhaps we try moving them to the pdb namespace later.

// this is a computation that applies a lambda to a tuple set
struct ApplyLambda : public AtomicComputation {

 private:
  std::string lambdaName;

 public:
  ~ApplyLambda() {}

  /**
 * A constructor to create the ApplyLambda class
 * @param input - the input tuple spec
 * @param output - the output tuple spec
 * @param projection - the projection
 * @param nodeName - the node name
 * @param lambdaNameIn - the name of the lambda
 */
  ApplyLambda(TupleSpec &input,
              TupleSpec &output,
              TupleSpec &projection,
              std::string nodeName,
              std::string lambdaNameIn)
      : AtomicComputation(input, output, projection, nodeName), lambdaName(lambdaNameIn) {}

  /**
   * A constructor to create the ApplyLambda class. This one also accepts the key value pairs that contain additional
   * information about hte l
   * @param input - the input tuple spec
   * @param output - the output tuple spec
   * @param projection - the projection
   * @param nodeName - the node name
   * @param lambdaNameIn - the name of the lambda
   * @param useMe - the info key value pairs
   */
  ApplyLambda(TupleSpec &input,
              TupleSpec &output,
              TupleSpec &projection,
              std::string nodeName,
              std::string lambdaNameIn,
              KeyValueList &useMe) : AtomicComputation(input, output, projection, nodeName),
                                     lambdaName(lambdaNameIn) {
    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  std::string getAtomicComputationType() override {
    return std::string("Apply");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return ApplyLambdaTypeID;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // the output from the apply is:
    //
    // (set of projection atts) (new attribute created from apply)
    //
    // find where the attribute appears in the outputs
    int counter = findPosInOutputAtts(attName);

    // if the attribute we are asking for is at the end (where the result of the lambda
    // application goes)
    // then we asked for it
    if (counter == getOutput().getAtts().size() - 1) {
      return std::make_pair(getComputationName(), lambdaName);
    }

    // otherwise, find our parent
    return allComps.getProducingAtomicComputation(getProjection().getSetName())
        ->findSource((getProjection().getAtts())[counter], allComps);
  }

  // returns the name of the lambda we are supposed to apply
  std::string &getLambdaToApply() {
    return lambdaName;
  }

  // wheter this lambda is a key extraction lambda
  bool isExtractingKey() {

    // get the lambda type
    auto it = keyValuePairs->find("lambdaType");
    return it != keyValuePairs->end() && it->second == "key";
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= APPLY (" << input <<  ", " << projection << ", '" << computationName  << "', '" << lambdaName << "')\n";
    return os;
  }
};

// this is a computation that applies a hash to a particular attribute in a tuple set
struct HashLeft : public AtomicComputation {

 private:
  std::string lambdaName;

 public:
  ~HashLeft() = default;

  HashLeft(TupleSpec &input,
           TupleSpec &output,
           TupleSpec &projection,
           std::string nodeName,
           std::string lambdaNameIn)
      : AtomicComputation(input, output, projection, nodeName), lambdaName(lambdaNameIn) {}

  // ss107: New Constructor:
  HashLeft(TupleSpec &input,
           TupleSpec &output,
           TupleSpec &projection,
           std::string nodeName,
           std::string lambdaNameIn,
           KeyValueList &useMe) : AtomicComputation(input, output, projection, nodeName), lambdaName(lambdaNameIn) {

    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  std::string getAtomicComputationType() override {
    return std::string("HashLeft");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return HashLeftTypeID;
  }

  // returns the name of the lambda we are supposed to apply
  std::string &getLambdaToApply() {
    return lambdaName;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // The output from the hash should be
    //
    // (projection atts) (hash value)
    //

    // find where the attribute appears in the outputs
    int counter = findPosInOutputAtts(attName);

    // if the attribute we are asking for is at the end (where the result of the lambda
    // application goes)
    // then we asked for it
    if (counter == getOutput().getAtts().size() - 1) {
      std::cout << "Why are you trying to find the origin of a hash value??\n";
      std::cout << "The attribute is " << attName << '\n';
      exit(1);
    }

    // otherwise, find our parent
    return allComps.getProducingAtomicComputation(getProjection().getSetName())
        ->findSource((getProjection().getAtts())[counter], allComps);
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= HASHLEFT (" << input <<  ", " << projection << ", '" << computationName << "', '" << lambdaName << "')\n";
    return os;
  }
};

// this is a computation that applies a lambda to a tuple set
struct HashRight : public AtomicComputation {

 private:
  std::string lambdaName;

 public:
  ~HashRight() = default;

  HashRight(TupleSpec &input,
            TupleSpec &output,
            TupleSpec &projection,
            std::string nodeName,
            std::string lambdaNameIn)
      : AtomicComputation(input, output, projection, nodeName), lambdaName(lambdaNameIn) {}

  // ss107: New Constructor:
  HashRight(TupleSpec &input,
            TupleSpec &output,
            TupleSpec &projection,
            std::string nodeName,
            std::string lambdaNameIn,
            KeyValueList &useMe) :
      AtomicComputation(input, output, projection, nodeName), lambdaName(lambdaNameIn) {

    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  std::string getAtomicComputationType() override {
    return std::string("HashRight");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return HashRightTypeID;
  }

  // returns the name of the lambda we are supposed to apply
  std::string &getLambdaToApply() {
    return lambdaName;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // The output from the hash should be
    //
    // (projection atts) (hash value)
    //

    // find where the attribute appears in the outputs
    int counter = findPosInOutputAtts(attName);

    // if the attribute we are asking for is at the end (where the result of the lambda
    // application goes) then we asked for it
    if (counter == getOutput().getAtts().size() - 1) {
      std::cout << "Why are you trying to find the origin of a hash value??\n";
      std::cout << "The attribute is " << attName << '\n';
      exit(1);
    }

    // otherwise, find our parent
    return allComps.getProducingAtomicComputation(getProjection().getSetName())
        ->findSource((getProjection().getAtts())[counter], allComps);
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= HASHRIGHT (" << input <<  ", " << projection << ", '" << computationName << "', '" << lambdaName << "')\n";
    return os;
  }
};

// this is a computation that adds 1  to each tuple of a tuple set
struct HashOne : public AtomicComputation {

 public:
  ~HashOne() override = default;

  HashOne(TupleSpec &input, TupleSpec &output, TupleSpec &projection, std::string nodeName) : AtomicComputation(input,
                                                                                                                output,
                                                                                                                projection,
                                                                                                                nodeName) {}

  // ss107: New Constructor:
  HashOne(TupleSpec &input, TupleSpec &output, TupleSpec &projection, std::string nodeName, KeyValueList &useMe) :
      AtomicComputation(input, output, projection, nodeName) {

    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  std::string getAtomicComputationType() override {
    return std::string("HashOne");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return HashOneTypeID;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // The output from the hash should be
    //
    // (projection atts) (hash value)
    //

    // find where the attribute appears in the outputs
    int counter = findPosInOutputAtts(attName);

    // otherwise, find our parent
    return allComps.getProducingAtomicComputation(getProjection().getSetName())
        ->findSource((getProjection().getAtts())[counter], allComps);
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= HASHONE (" << input <<  ", " << projection << ", '" << computationName << "')\n";
    return os;
  }
};

// this is a computation that flatten each tuple of a tuple set
struct Flatten : public AtomicComputation {
public:

  ~Flatten() = default;

  Flatten(TupleSpec &input, TupleSpec &output, TupleSpec &projection, std::string nodeName)
      : AtomicComputation(input,
                          output,
                          projection,
                          nodeName) {}

  // ss107: New Constructor:
  Flatten(TupleSpec &input,
          TupleSpec &output,
          TupleSpec &projection,
          std::string nodeName,
          KeyValueList &useMe) : AtomicComputation(input,
                                                   output,
                                                   projection,
                                                   nodeName) {
    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  std::string getAtomicComputationType() override {
    return std::string("Flatten");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return FlattenTypeID;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // The output from the hash should be
    //
    // (projection atts) (hash value)
    //
    // std :: cout << "Flatten findSource for attName=" << attName << std :: endl;
    // find where the attribute appears in the outputs
    int counter = findPosInOutputAtts(attName);

    if (counter == getOutput().getAtts().size() - 1) {
      return std::make_pair(getComputationName(), std::string(""));
    }

    return allComps.getProducingAtomicComputation(getProjection().getSetName())
        ->findSource((getProjection().getAtts())[counter], allComps);
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= FLATTEN (" << input <<  ", " << projection << ", '" << computationName << "')\n";
    return os;
  }
};

// this is a computation that unions two tuple sets
// this is a computation that flatten each tuple of a tuple set
struct Union : public AtomicComputation {
 private:

  TupleSpec rightInput;

 public:

  Union(TupleSpec &output, TupleSpec &lhsInput, TupleSpec &rhsInput, std::string nodeName)
      : AtomicComputation(lhsInput,
                          output,
                          lhsInput,
                          std::move(nodeName)) {
    rightInput = rhsInput;
  }

  // ss107: New Constructor:
  Union(TupleSpec &output,
        TupleSpec &lhsInput,
        TupleSpec &rhsInput,
        std::string nodeName,
        KeyValueList &useMe) : AtomicComputation(lhsInput,
                                                 output,
                                                 TupleSpec(), // we don't keep any of the attributes from the input sets
                                                 std::move(nodeName)) {
    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();

    // set the rhs input
    rightInput = rhsInput;
  }

  ~Union() override = default;

  std::string getAtomicComputationType() override {
    return std::string("Union");
  }

  TupleSpec &getRightInput() override {
    return rightInput;
  }

  // the projection is the same as the input
  TupleSpec &getRightProjection() override {
    return rightInput;
  }

  bool hasTwoInputs() override {
    return true;
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return UnionTypeID;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // The output from the union should be a single attribute
    // find where the attribute appears in the outputs
    int counter = findPosInOutputAtts(attName);

    // if the attribute we are asking for is at the end, it means it's produced by this
    // aggregate then we asked for it
    if (counter == 0) {
      return std::make_pair(getComputationName(), std::string(""));
    }

    // if it is not at the end, if makes no sense
    std::cout << "How did we ever get here trying to find an attribute produced by an agg??\n";
    exit(1);
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= UNION (" << input <<  ", " << rightInput << ", '" << computationName << "')\n";
    return os;
  }
};

// this is a computation that performs a filer over a tuple set
struct ApplyFilter : public AtomicComputation {

 public:
  ApplyFilter(TupleSpec &input, TupleSpec &output, TupleSpec &projection, std::string nodeName)
      : AtomicComputation(input, output, projection, nodeName) {
    // std :: cout << "Filter input tuple spec: " << input << ", output tuple spec: " << output
    // << ", projection tuple spec: " << projection << std :: endl;
  }

  // ss107: New Constructor:
  ApplyFilter(TupleSpec &input, TupleSpec &output, TupleSpec &projection, std::string nodeName, KeyValueList &useMe)
      : AtomicComputation(input, output, projection, nodeName) {

    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  ~ApplyFilter() override = default;

  std::string getAtomicComputationType() override {
    return std::string("Filter");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return ApplyFilterTypeID;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // the output from the filter should be identical to the set of projection attributes
    // find where the attribute appears in the outputs
    int counter = findPosInOutputAtts(attName);

    // otherwise, find our parent
    return allComps.getProducingAtomicComputation(getProjection().getSetName())
        ->findSource((getProjection().getAtts())[counter], allComps);
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= FILTER (" << input <<  ", " << projection << ", '" << computationName << "')\n";
    return os;
  }
};

// this is a computation that aggregates a tuple set
struct ApplyAgg : public AtomicComputation {

public:

  ~ApplyAgg() override = default;

  ApplyAgg(TupleSpec &input, TupleSpec &output, TupleSpec &projection, std::string nodeName)
      : AtomicComputation(input, output, projection, nodeName) {}

  // ss107: New Constructor:
  ApplyAgg(TupleSpec &input, TupleSpec &output, TupleSpec &projection, std::string nodeName, KeyValueList &useMe) :
      AtomicComputation(input, output, projection, nodeName) {

    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  std::string getAtomicComputationType() override {
    return std::string("Aggregate");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return ApplyAggTypeID;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // The output from the aggregate should be a single attribute
    // find where the attribute appears in the outputs
    int counter = findPosInOutputAtts(attName);

    // if the attribute we are asking for is at the end, it means it's produced by this
    // aggregate then we asked for it
    if (counter == 0) {
      return std::make_pair(getComputationName(), std::string(""));
    }

    // if it is not at the end, if makes no sense
    std::cout << "How did we ever get here trying to find an attribute produced by an agg??\n";
    exit(1);
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= AGGREGATE ( '" << input <<  ", '" << computationName << "')\n";
    return os;
  }
};

// this is a computation that produces a tuple set by scanning a set stored in the database
struct ScanSet : public AtomicComputation {
protected:

  // the specifier of the set (database, set)
  std::pair<std::string, std::string> setSpecifier;

  // The page set identifier (taskID, specifier)
  std::pair<size_t, std::string> pageSetSpecifier;

public:

  ~ScanSet() override = default;

  ScanSet(TupleSpec &output, const std::string &dbName, const std::string &setName, const std::string &nodeName)
      : AtomicComputation(TupleSpec(), output, TupleSpec(), nodeName),
        setSpecifier(std::make_pair(dbName, setName)) {}

  // ss107: New Constructor:
  ScanSet(TupleSpec &output, const std::string &dbName, const std::string &setName, const std::string &nodeName, KeyValueList &useMe) :
      AtomicComputation(TupleSpec(), output, TupleSpec(), nodeName),
      setSpecifier(std::make_pair(dbName, setName)) {

    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  ScanSet(TupleSpec &output, size_t taskID, const std::string &pageSetSpecifier, const std::string &nodeName, KeyValueList &useMe) :
    AtomicComputation(TupleSpec(), output, TupleSpec(), nodeName), pageSetSpecifier(std::make_pair(taskID, pageSetSpecifier)) {

    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  ScanSet(TupleSpec &output, size_t taskID, const std::string &pageSetSpecifier, const std::string &nodeName) :
      AtomicComputation(TupleSpec(), output, TupleSpec(), nodeName), pageSetSpecifier(std::make_pair(taskID, pageSetSpecifier)) {}

  std::string getAtomicComputationType() override {
    return std::string("Scan");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return ScanSetAtomicTypeID;
  }

  std::string &getDBName() {
    return setSpecifier.first;
  }

  std::string &getSetName() {
    return setSpecifier.second;
  }

  std::pair<size_t, std::string> &getPageSetSpecifier() {
    return pageSetSpecifier;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // The output from the scan should be a single attribute
    // find where the attribute appears in the outputs
    int counter = findPosInOutputAtts(attName);

    // if the attribute we are asking for is at the end (where the result of the lambda
    // application goes)
    // then we asked for it
    if (counter == 0) {
      return std::make_pair(getComputationName(), std::string(""));
    }

    // if it is not at the end, if makes no sense
    std::cout
        << "How did we ever get here trying to find an attribute produced by a scan set??\n";
    exit(1);
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= SCAN ( '" << setSpecifier.first << "', '" << setSpecifier.second << "', '" << computationName << "')\n";
    return os;
  }
};

// this is a computation that writes out a tuple set
struct WriteSet : public AtomicComputation {

  std::string dbName;
  std::string setName;

 public:
  ~WriteSet() = default;

  WriteSet(TupleSpec &input,
           TupleSpec &output,
           TupleSpec &projection,
           std::string dbName,
           std::string setName,
           std::string nodeName)
      : AtomicComputation(input, output, projection, nodeName),
        dbName(dbName),
        setName(setName) {}

  // ss107: New Constructor:
  WriteSet(TupleSpec &input,
           TupleSpec &output,
           TupleSpec &projection,
           std::string dbName,
           std::string setName,
           std::string nodeName,
           KeyValueList &useMe) :
      AtomicComputation(input, output, projection, nodeName), dbName(dbName), setName(setName) {

    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  std::string getAtomicComputationType() override {
    return std::string("WriteSet");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return WriteSetTypeID;
  }

  std::string &getDBName() {
    return dbName;
  }

  std::string &getSetName() {
    return setName;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {
    std::cout << "How did we ever get to a write set trying to find an attribute??\n";
    exit(1);
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= OUTPUT ( " << input << ", '" << dbName << "', '" << setName << "', '" << computationName << "')\n";
    return os;
  }
};

struct ApplyJoin : public AtomicComputation {

  TupleSpec rightInput;

  // the right projection of the join (basically the inputs we keepo )
  TupleSpec rightProjection;

  // true if this is a key join false otherwise
  bool isKeyJoin = false;

  ApplyJoin(TupleSpec &output,
            TupleSpec &lInput,
            TupleSpec &rInput,
            TupleSpec &lProjection,
            TupleSpec &rProjection,
            const std::string &nodeName)
      : AtomicComputation(lInput, output, lProjection, nodeName),
        rightInput(rInput),
        rightProjection(rProjection) {}

  // ss107: New Constructor: Added Jia's correction too:
  ApplyJoin(TupleSpec &output,
            TupleSpec &lInput,
            TupleSpec &rInput,
            TupleSpec &lProjection,
            TupleSpec &rProjection,
            const std::string &nodeName,
            KeyValueList &useMe) :
      AtomicComputation(lInput, output, lProjection, nodeName), rightInput(rInput), rightProjection(rProjection) {

    // set the key value pairs
    keyValuePairs = useMe.getKeyValuePairs();
  }

  TupleSpec &getRightProjection() override {
    return rightProjection;
  }

  TupleSpec &getRightInput() override {
    return rightInput;
  }
  bool hasTwoInputs() override {
    return true;
  }

  std::string getAtomicComputationType() override {
    return std::string("JoinSets");
  }

  AtomicComputationTypeID getAtomicComputationTypeID() override {
    return ApplyJoinTypeID;
  }

  std::pair<std::string, std::string> findSource(std::string attName,
                                                 AtomicComputationList &allComps) override {

    // The output from the join should be
    //
    // (left projection atts) (right projection atts)
    //
    // so find where the attribute in question came from
    int counter = findPosInOutputAtts(attName);

    // if it came from the left, then we recurse and find it
    if (counter < getProjection().getAtts().size()) {
      return allComps.getProducingAtomicComputation(getProjection().getSetName())
          ->findSource((getProjection().getAtts())[counter], allComps);

      // otherwise, if it came from the right, recurse and find it
    } else if (counter < getProjection().getAtts().size() + rightProjection.getAtts().size()) {
      return allComps.getProducingAtomicComputation(rightProjection.getSetName())
          ->findSource(
              (rightProjection.getAtts())[counter - getProjection().getAtts().size()],
              allComps);

    } else {
      std::cout << "Why in the heck did we not find the producer when checking a join!!??\n";
      exit(1);
    }
  }

  // serializes this atomic computation
  std::ostream &writeOut(std::ostream &os) const override {
    os << output << " <= JOIN (" << input << ", " << projection << ", " << rightInput << ", " << rightProjection << ", '" << computationName << "')\n";
    return os;
  }
};