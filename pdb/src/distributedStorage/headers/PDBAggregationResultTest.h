#pragma once

/**
 * Here are the SFINAE test to check whether a value of a set is capable of being an aggregation result.
 * Basically it checks if it has two methods getKey and getValue
 */
namespace pdb {

// SFINAE test for getKey
template <typename T>
class hasGetKey
{
  typedef char one;
  typedef long two;

  template <typename C> static one test( typeof(&C::getKey) ) ;
  template <typename C> static two test(...);

 public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

// SFINAE test for getValue
template <typename T>
class hasGetValue
{
  typedef char one;
  typedef long two;

  template <typename C> static one test( typeof(&C::getValue) ) ;
  template <typename C> static two test(...);

 public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

}