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


#include <benchmark/benchmark.h>

#include "Handle.h"
#include "PDBVector.h"
#include "InterfaceFunctions.h"
#include "Employee.h"
#include "Supervisor.h"

using namespace pdb;

/**
 * The Employee class to bench std vector
 */
class EmployeeNative {

  std::string *name{nullptr};
  int age{};

 public:

  ~EmployeeNative() {
    delete name;
  }

  EmployeeNative() = default;

  EmployeeNative(const std::string &nameIn, int ageIn) {
    name = new std::string(nameIn);
    age = ageIn;
  }
};

/**
 * The Supervisor class to bench std vector
 */
class SupervisorNative {

 private:
  EmployeeNative *me{nullptr};
  std::vector<EmployeeNative *> myGuys;

 public:

  SupervisorNative() {
    for (auto a : myGuys) {
      delete a;
    }
  }

  ~SupervisorNative() = default;

  SupervisorNative(const std::string &name, int age) {
    me = new EmployeeNative(name, age);
  }

  void addEmp(EmployeeNative *addMe) {
    myGuys.push_back(addMe);
  }
};

static void BenchPDBVector(benchmark::State& state) {

  // how many objects
  const int NUM_OBJECTS = 12000;

  // bench
  for (auto _ : state) {

    // load up the allocator with RAM
    makeObjectAllocatorBlock(1024 * 1024 * 24, true);

    // create a vector
    Handle<Vector<Handle<Supervisor>>> supers = makeObject<Vector<Handle<Supervisor>>>(10);

    try {

      // put a lot of copies of it into a vector
      for (int i = 0; i < NUM_OBJECTS; i++) {

        // push the object
        supers->push_back(makeObject<Supervisor>("Joe Johnson", 20 + (i % 29)));

        // create 10 employee objects and push them
        for (int j = 0; j < 10; j++) {
          Handle<Employee> temp = makeObject<Employee>("Steve Stevens", 20 + ((i + j) % 29));
          (*supers)[supers->size() - 1]->addEmp(temp);
        }
      }


      supers->pop_back();

    } catch (NotEnoughSpace& e) {
      // so we are out of memory on the block finish
    }

    // do not optimize out the value
    benchmark::DoNotOptimize(supers);
  }
}

static void BenchNativeVector(benchmark::State& state) {

  // how many objects
  const int NUM_OBJECTS = 12000;

  // bench
  for (auto _ : state) {

    std::vector<SupervisorNative*> supers;

    // put a lot of copies of it into a vector
    for (int i = 0; i < NUM_OBJECTS; i++) {

      // push the object
      supers.push_back(new SupervisorNative("Joe Johnson", 20 + (i % 29)));

      // create 10 employee objects and push them
      for (int j = 0; j < 10; j++) {
        auto* temp = new EmployeeNative("Steve Stevens", 20 + ((i + j) % 29));
        supers[supers.size() - 1]->addEmp(temp);
      }
    }

    // do not optimize out the value
    benchmark::DoNotOptimize(supers);
  }
}

// Register the function as a benchmark
BENCHMARK(BenchPDBVector);
BENCHMARK(BenchNativeVector);

// create the main function
BENCHMARK_MAIN();