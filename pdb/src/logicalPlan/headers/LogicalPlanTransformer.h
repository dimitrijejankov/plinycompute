#pragma once

#include "LogicalPlan.h"

namespace pdb {

// base class of every transformation
class Transformation {
public:

  // the plan we want to set
  void setPlan(LogicalPlanPtr planToSet);

  // apply the transformation
  virtual void apply() = 0;

  // can we actually apply the transformation
  virtual bool canApply() = 0;

  // we drop all dependencies of the tuple set
  void dropDependencies(const std::string &tupleSetName);

  // we drop all dependents of the tuple set
  void dropDependents(const std::string &tupleSetName);

protected:

  // the logical plan we are transforming
  LogicalPlanPtr logicalPlan;
};

// the shared ptr version of the transformation
using TransformationPtr = std::shared_ptr<Transformation>;


// applies a bunch of transformations to the logical plan
class LogicalPlanTransformer {
public:

  explicit LogicalPlanTransformer(LogicalPlanPtr &logicalPlan);

  // adds a transformation to the transformer
  void addTransformation(const TransformationPtr &transformation);

  // apply app the transformations to the plan
  LogicalPlanPtr applyTransformations();

private:

  // the transformations we are going to apply
  std::vector<TransformationPtr> transformationsToApply;

  // the logical plan we are transforming
  LogicalPlanPtr logicalPlan;
};

/**
 * InsertKeyScanSetsTransformation
 */
class InsertKeyScanSetsTransformation : public Transformation {
public:

  explicit InsertKeyScanSetsTransformation(std::string inputTupleSet);

  void apply() override;
  bool canApply() override;

private:

  // the input tuple set
  std::string inputTupleSet;

  // the identifier of the page set
  std::pair<size_t, std::string> pageSetIdentifier;

};

/**
 * JoinFromKeyTransformation
 */

class JoinKeySideTransformation : public Transformation {
 public:

  explicit JoinKeySideTransformation(std::string inputTupleSet);

  void apply() override;
  bool canApply() override;

 private:

  // the input tuple set
  std::string inputTupleSet;

};

/**
 * JoinKeyTransformation
 */

class JoinKeyTransformation : public Transformation {
 public:

  explicit JoinKeyTransformation(std::string joinTupleSet);

  void apply() override;
  bool canApply() override;

 private:

  // the input tuple set
  std::string joinTupleSet;

};

/**
 * JoinKeyTransformation
 */


class AggKeyTransformation : public Transformation {
 public:

  explicit AggKeyTransformation(std::string aggStartTupleSet);

  void apply() override;
  bool canApply() override;

 private:

  // the input tuple set
  std::string aggStartTupleSet;

};

/**
 * Drop Dependents
 */


class DropDependents : public Transformation {
 public:

  explicit DropDependents(const string startTupleSet);

  void apply() override;
  bool canApply() override;

 private:

  // the input tuple set
  std::string startTupleSet;

};

/**
 * Drop AddJoinTID
 */


class AddJoinTID : public Transformation {
 public:

  explicit AddJoinTID(std::string joinTupleSet);

  void apply() override;
  bool canApply() override;

 private:

  // the input tuple set
  std::string joinTupleSet;

};


}
