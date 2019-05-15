//
// Created by dimitrije on 3/29/19.
//

#pragma once

#include <queue>
#include <condition_variable>

template <typename T>
class concurent_queue {

private:

  // the queue implementation
  std::queue<T> _internal_queue;

  // the mutex to lock the structure
  std::mutex _m;

  // the conditional variable to wait
  std::condition_variable _cv;

public:

  inline void wait_dequeue(T &item) {

    // wait until we have something in the queue
    std::unique_lock<std::mutex> lk(_m);
    _cv.wait(lk, [&]{return !_internal_queue.empty();});

    // grab the element and pop the queue
    item = _internal_queue.front();
    _internal_queue.pop();
  };

  inline void enqueue(T const& item) {

    // wait for lock
    std::unique_lock<std::mutex> lk(_m);

    // insert the element in the queue
    _internal_queue.push(item);

    // notify all the threads that are waiting
    _cv.notify_all();
  }

};
