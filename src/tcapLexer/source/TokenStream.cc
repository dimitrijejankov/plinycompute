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
#include <iostream>
#include "TokenStream.h"

namespace pdb_detail
{

    TokenStream::TokenStream(shared_ptr<vector<Token>>tokens) :_tokens(tokens)
    {
    }

    bool TokenStream::hasNext()
    {
        return _readIndex < _tokens->size();
    }

    Token TokenStream::advance()
    {
        if(_readIndex>=_tokens->size())
            return Token("", TokenType::UNKNOWN_TYPE);

        return _tokens->operator[](_readIndex++);
    }

    Token TokenStream::peek()
    {
        return _tokens->operator[](_readIndex);
    }

// useful for debugging.
//
//        void TokenStream::printTypes()
//        {
//            for(int i = _readIndex; i<_tokens->size(); i++)
//            {
//                std::cerr << _tokens->operator[](i).tokenType << "\n";
//            }
//        }

}