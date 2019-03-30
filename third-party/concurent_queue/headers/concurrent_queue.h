//
// Created by dimitrije on 3/29/19.
//

#pragma once

#include <concurrent_queue_internal.h>
#include <blocking_concurrent_queue_internal.h>

template<typename T, typename Traits = moodycamel::ConcurrentQueueDefaultTraits>
using concurent_queue = moodycamel::ConcurrentQueue<T, Traits>;

template<typename T, typename Traits = moodycamel::ConcurrentQueueDefaultTraits>
using blocking_concurent_queue = moodycamel::BlockingConcurrentQueue<T, Traits>;
