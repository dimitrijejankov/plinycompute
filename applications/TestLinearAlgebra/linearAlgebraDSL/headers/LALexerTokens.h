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
#ifndef LA_LEX_TOKENS_H
#define LA_LEX_TOKENS_H

// by Binhang, June 2017

#ifdef END
#undef END
#undef TOKEN_ZEROS
#undef TOKEN_ONES
#undef TOKEN_IDENTITY
#undef TOKEN_LOAD
#undef TOKEN_TRANSPOSE
#undef TOKEN_INV
#undef TOKEN_MULTIPLY
#undef TOKEN_TRANSPOSE_MULTIPLY
#undef TOKEN_ELEMENTWISE_MULTIPLY
#undef TOKEN_ADD
#undef TOKEN_MINUS
#undef TOKEN_LEFT_BRACKET
#undef TOKEN_RIGHT_BRACKET
#undef TOKEN_COMMA
#undef TOKEN_ASSIGN
#undef TOKEN_MAX
#undef TOKEN_MIN
#undef TOKEN_MAX
#undef TOKEN_MIN
#undef TOKEN_ROW_MAX
#undef TOKEN_ROW_MIN
#undef TOKEN_ROW_SUM
#undef TOKEN_COL_MAX
#undef TOKEN_COL_MIN
#undef TOKEN_COL_SUM
#undef TOKEN_DUPLICATE_ROW
#undef TOKEN_DUPLICATE_COL
#undef TOKEN_DOUBLE
#undef TOKEN_INTEGER
#undef IDENTIFIER_LITERAL
#undef STRING_LITERAL
#endif


#define END 0
// Initializer
#define TOKEN_ZEROS 701
#define TOKEN_ONES 702
#define TOKEN_IDENTITY 703
#define TOKEN_LOAD 704

// postfix operator
#define TOKEN_TRANSPOSE 711
#define TOKEN_INV 712

// multiplicative operator
#define TOKEN_MULTIPLY 721
#define TOKEN_TRANSPOSE_MULTIPLY 722
#define TOKEN_ELEMENTWISE_MULTIPLY 723

// additive operator
#define TOKEN_ADD 731
#define TOKEN_MINUS 732

// Others
#define TOKEN_LEFT_BRACKET 741
#define TOKEN_RIGHT_BRACKET 742
#define TOKEN_COMMA 743
#define TOKEN_ASSIGN 744

// Built in functions
#define TOKEN_MAX 751
#define TOKEN_MIN 752
#define TOKEN_ROW_MAX 753
#define TOKEN_ROW_MIN 754
#define TOKEN_ROW_SUM 755
#define TOKEN_COL_MAX 756
#define TOKEN_COL_MIN 757
#define TOKEN_COL_SUM 758
#define TOKEN_DUPLICATE_ROW 759
#define TOKEN_DUPLICATE_COL 760


#define TOKEN_DOUBLE 790
#define TOKEN_INTEGER 791
#define IDENTIFIER_LITERAL 792
#define STRING_LITERAL 793

#endif