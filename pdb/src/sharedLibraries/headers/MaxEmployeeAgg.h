//
// Created by vicram on 6/25/19.
//

#ifndef PDB_MAXEMPLOYEEAGG_H
#define PDB_MAXEMPLOYEEAGG_H

#include <Employee.h>
#include <AggregateComp.h>
#include <DepartmentMax.h>
#include <LambdaCreationFunctions.h>
#include <cmath>

namespace pdb {

/**
 * This "adder" takes an int and a double, rounds the double, and returns the greater of the two numbers.
 * This will be the "VTAdder" type for the Aggregation.
 */
class IntDoubleMax : public AbstractAdder<int, double> {
 public:
  int add(int& in1, double& in2) override {
    // First, convert the double
    int rounded = std::lround(in2);
    return (in1 >= rounded) ? in1 : rounded;
  }
};

/**
 * This "adder" returns the greater of two ints. It will be the "VVAdder" for the Aggregation.
 */
class IntIntMax : public AbstractAdder<int> {
 public:
  int add(int& in1, int& in2) override {
    return (in1 >= in2) ? in1 : in2;
  }
};

/**
 * This converter rounds a double to an int.
 */
class DoubleToIntConverter : public AbstractConverter<double, int> {
 public:
  void convert(double& in, int* out) override {
    int rounded = std::lround(in);
    *out = rounded;
  }
};

/**
 * This aggregation finds, for each department, the highest salary among all that
 * department's Employees, rounded to the nearest dollar.
 */
class MaxEmployeeAgg : public AggregateComp<
    DepartmentMax, // Output
    Employee, // Input
    String, // Key
    int, // Value
    double, // TempValue
    IntDoubleMax, // VTAdder
    IntIntMax, //VVAdder
    DoubleToIntConverter> { // Converter
 public:
  ENABLE_DEEP_COPY

  Lambda<String> getKeyProjection(Handle<Employee> aggMe) override {
    return makeLambdaFromMember (aggMe, department);
  }

  Lambda<double> getValueProjection(Handle<Employee> aggMe) override {
    return makeLambdaFromMethod (aggMe, getSalary);
  }
};

}

#endif //PDB_MAXEMPLOYEEAGG_H
