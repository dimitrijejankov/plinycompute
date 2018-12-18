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

#define NUM_OBJECTS 10371

#include <benchmark/benchmark.h>

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

    SupervisorNative(std::string name, int age) {
        me = new EmployeeNative(name, age);
    }

    void addEmp(EmployeeNative *addMe) {
        myGuys.push_back(addMe);
    }
};

static void BenchPDBVectorClear(benchmark::State &state) {

    // bench
    for (auto _ : state) {

        // pause the timing
        state.PauseTiming();

        // load up the allocator with RAM
        makeObjectAllocatorBlock(1024 * 1024 * 24, true);

        Handle<Vector<Handle<Supervisor>>> supers = makeObject<Vector<Handle<Supervisor>>>();
        try {

            // put a lot of copies of it into a vector
            for (int i = 0; i < NUM_OBJECTS; i++) {

                Handle<Supervisor> super = makeObject<Supervisor>("Joe Johnson", 20 + (i % 29));
                supers->push_back(super);
                for (int j = 0; j < 10; j++) {
                    Handle<Employee> temp = makeObject<Employee>("Steve Stevens", 20 + ((i + j) % 29));
                    (*supers)[i]->addEmp(temp);
                }
            }

        } catch (NotEnoughSpace &e) {}

        // resume timing
        state.ResumeTiming();

        // clear them
        supers->clear();

        // do not optimize out the value
        benchmark::DoNotOptimize(supers);
    }
}

// Register the function as a benchmark
BENCHMARK(BenchPDBVectorClear);

// create the main function
BENCHMARK_MAIN();