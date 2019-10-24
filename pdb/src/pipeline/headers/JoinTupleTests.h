#pragma once

#include <type_traits>

namespace pdb {

// this needs to be moved to some place it actually makes sense
namespace tupleTests {

// test to checki if the join tuple has getKey method

template<class>
struct sfinae_true : std::true_type{};

template<class T>
static auto test_get_key(int) -> sfinae_true<decltype(std::declval<T>().getKey())>;
template<class>
static auto test_get_key(long) -> std::false_type;

template<class T>
struct has_get_key : decltype(test_get_key<T>(0)){};

}

}