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
#include "InterfaceFunctions.h"
#include "PDBString.h"

using namespace pdb;

static void BenchPDBVectorWithCopy(benchmark::State& state) {

    // bench
    for (auto _ : state) {

        // load up the allocator with RAM
        makeObjectAllocatorBlock(1024 * 1024 * 24, false);

        for (int i = 0; i < 10000; i++) {
            Handle<String> str = makeObject<String>(
                "This is an object big enough to force flushing soon. This is an object big enough to "
                "force flushing soon. This is an object big enough to force flushing soon. This is an "
                "object big enough to force flushing soon. This is an object big enough to force "
                "flushing soon. This is an object big enough to force flushing soon. This is an object "
                "big enough to force flushing soon. This is an object big enough to force flushing "
                "soon. This is an object big enough to force flushing  soon. It has a total of 512 "
                "bytes to test. This is an object big enough to force flushing soon. This is an object "
                "big enough to force flushing soon. This is an object big enough to force flushing "
                "soon. This is an object big enough to force flushing soon. This is an object big "
                "enough to force flushing..");

            // do not optimize out the value
            benchmark::DoNotOptimize(str);
        }
    }
}

// Register the function as a benchmark
BENCHMARK(BenchPDBVectorWithCopy);

// create the main function
BENCHMARK_MAIN();