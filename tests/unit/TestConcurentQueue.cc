//
// Created by dimitrije on 5/15/19.
//

#include <thread>
#include <vector>
#include <concurrent_queue.h>
#include <atomic>
#include <iostream>
#include <gtest/gtest.h>

TEST(TestConcurentQueue, Test) {

  const int numThreads = 10;
  const int range = 10000;
  concurent_queue<int> tmp;

  std::atomic<int> counter;
  counter = 0;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for(int t = 0; t < numThreads; ++t) {
    for(int i = 0; i < range; i++) {
      tmp.enqueue(i);
    }

    counter++;
    if(counter == numThreads) {
      tmp.enqueue(-1);
    }
  }

  int64_t sum = 0;
  while(true) {
    int val;
    tmp.wait_dequeue(val);

    // finish on -1
    if(val == -1) {
      break;
    }

    // increment the sum
    sum += val;
  }

  // check the result
  EXPECT_EQ(sum, numThreads * (((range - 1) * range) / 2));

  for(auto &t : threads) {
    t.join();
  }
}