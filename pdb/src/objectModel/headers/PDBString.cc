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

#ifndef PDBSTRING_CC
#define PDBSTRING_CC

#include "PDBString.h"
#include <string.h>
#include <iostream>
#include <string>
#include "PDBMap.h"
#include "BuiltInObjectTypeIDs.h"

#define CAN_FIT_IN_DATA(len) (len <= sizeof(decltype(data().getOffset())))

namespace pdb {

inline Handle<char>& String::data() const {
    return *((Handle<char>*)this->storage);
}

inline String::String() {

    // view the Handle as actually storing data
    data().setExactTypeInfoValue(1);
    data().setOffset(-1);
    c_str()[0] = 0;
}

// does a bunch of bit twiddling to compute a hash of an int
inline unsigned int newHash2(unsigned int x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}


// computes a hash over an input character array
inline size_t hashMe2(char* me, size_t len) {
  size_t code = 0;
#ifndef HASH_FOR_TPCH
  for (int i = 0; i < len; i += 8) {

    int low = 0;
    int high = 0;

    for (int j = 0; j < 4; j++) {
      if (i + j < len) {
        low += (me[i + j] & 0xFF) << (8 * j);
      } else {
        low += 123 << (8 * j);
      }
    }

    if (len <= 4)
      high = low;
    else {
      for (int j = 4; j < 8; j++) {
        if (i + j < len) {
          high += (me[i + j] & 0xFF) << (8 * (j - 3));
        } else {
          high += 97 << (8 * (j - 3));
        }
      }
    }

    size_t returnVal = ((size_t)newHash2(high)) << 32;
    returnVal += newHash2(low) & 0xFFFFFFFF;
    code = code ^ returnVal;
  }
#else
  for (int i = 0; i < len; i++) {
        code = 31 * code + me[i];
    }

#endif
  return code;
}


inline size_t String::hash() const {

    return hashMe2(c_str(), size() - 1);
}

inline String& String::operator=(const char* toMe) {
    int len = strlen(toMe) + 1;
    if (CAN_FIT_IN_DATA(len)) {
        if (data().getExactTypeInfoValue() < 0) {
            data() = nullptr;
        }
        data().setExactTypeInfoValue(len);
    } else {
        if (data().getExactTypeInfoValue() >= 0) {
            data().setOffset(-1);
        }
        data() = makeObjectWithExtraStorage<char>((len - 1) * sizeof(char));
        data().setExactTypeInfoValue(-len);
    }

    memmove(c_str(), toMe, len);
    return *this;
}

inline String::String(const String& s) {

    // if the other guy is short
    if (s.data().getExactTypeInfoValue() >= 0) {
        data().setOffset(s.data().getOffset());

        // the other guy is big
    } else {
        data().setOffset(-1);
        data() = s.data();
    }

    data().setExactTypeInfoValue(s.data().getExactTypeInfoValue());
}

inline String::~String() {
    if (data().getExactTypeInfoValue() < -1) {
        (&data())->~Handle();
    }
}

inline String& String::operator=(const String& s) {

    if (this == &s) {
        std::cout << "Assigning to myself!!!!\n";
        exit(1);
    }

    // if the other guy is short
    if (s.data().getExactTypeInfoValue() >= 0) {
        if (data().getExactTypeInfoValue() < 0) {
            data() = nullptr;
        }
        data().setOffset(s.data().getOffset());

        // the other guy is big
    } else {
        if (data().getExactTypeInfoValue() >= 0) {
            data().setOffset(-1);
        }
        data() = s.data();
    }
    data().setExactTypeInfoValue(s.data().getExactTypeInfoValue());
    return *this;
}

inline String& String::operator=(const std::string& s) {
    int len = s.size() + 1;
    if (CAN_FIT_IN_DATA(len)) {
        if (data().getExactTypeInfoValue() < 0) {
            data() = nullptr;
        }
        data().setExactTypeInfoValue(len);
    } else {
        if (data().getExactTypeInfoValue() >= 0) {
            data().setOffset(-1);
        }
        data() = makeObjectWithExtraStorage<char>((len - 1) * sizeof(char));
        data().setExactTypeInfoValue(-len);
    }

    memmove(c_str(), s.c_str(), len);
    return *this;
}

inline String::String(const char* toMe) {
    int len = strlen(toMe) + 1;
    if (CAN_FIT_IN_DATA(len)) {
        data().setExactTypeInfoValue(len);
    } else {
        data().setOffset(-1);
        data() = makeObjectWithExtraStorage<char>((len - 1) * sizeof(char));
        data().setExactTypeInfoValue(-len);
    }

    memmove(c_str(), toMe, len);
}

inline String::String(const char* toMe, size_t n) {
    int len = n + 1;

    if (CAN_FIT_IN_DATA(len)) {
        data().setExactTypeInfoValue(len);
    } else {
        data().setOffset(-1);
        data() = makeObjectWithExtraStorage<char>((len - 1) * sizeof(char));
        data().setExactTypeInfoValue(-len);
    }

    memmove(c_str(), toMe, len - 1);
    c_str()[len - 1] = 0;
}

inline String::String(const std::string& s) {
    int len = s.size() + 1;

    if (CAN_FIT_IN_DATA(len)) {
        data().setExactTypeInfoValue(len);
    } else {
        data().setOffset(-1);
        data() = makeObjectWithExtraStorage<char>((len - 1) * sizeof(char));
        data().setExactTypeInfoValue(-len);
    }

    memmove(c_str(), s.c_str(), len);
}

inline char& String::operator[](int whichOne) {
    return c_str()[whichOne];
}

inline String::operator std::string() const {
    return std::string(c_str());
}

inline char* String::c_str() const {

    if (data().getExactTypeInfoValue() >= 0) {
        return (char*)&data();
    } else {
        return &(*data());
    }
}

inline size_t String::size() const {
    if (data().getExactTypeInfoValue() >= 0) {
        return data().getExactTypeInfoValue();
    } else {
        return -data().getExactTypeInfoValue();
    }
}

inline std::ostream& operator<<(std::ostream& stream, const String& printMe) {
    char* returnVal = printMe.c_str();
    stream << returnVal;
    return stream;
}

inline bool String::operator==(const String& toMe) {
    return strcmp(c_str(), toMe.c_str()) == 0;
}

inline bool String::operator==(const std::string& toMe) {
    return strcmp(c_str(), toMe.c_str()) == 0;
}

inline bool String::operator==(const char* toMe) {
    return strcmp(c_str(), toMe) == 0;
}

inline bool String::operator!=(const String& toMe) {
    return strcmp(c_str(), toMe.c_str()) != 0;
}

inline bool String::operator!=(const std::string& toMe) {
    return strcmp(c_str(), toMe.c_str()) != 0;
}

inline bool String::operator!=(const char* toMe) {
    return strcmp(c_str(), toMe) != 0;
}


inline bool String::endsWith(const std::string& suffix) {
    std::string str = c_str();
    int len1 = str.length();
    int len2 = suffix.length();
    if (len1 > len2) {
         return (str.compare(len1 - len2, len2, suffix) == 0);
    } else {
         return false;
    }
}

}

#endif

