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

#ifndef LEX_TOKENS_H
#define LEX_TOKENS_H

#ifdef FILTER
#undef FILTER
#undef APPLY
#undef SCAN
#undef AGG
#undef JOIN
#undef OUTPUT
#undef GETS
#undef IDENTIFIER
#undef STRING
#undef HASHLEFT
#undef HASHRIGHT
#undef HASHONE
#undef FLATTEN
#undef PARTITION
#undef UNION
#endif

#define FILTER 258
#define APPLY 259
#define SCAN 260
#define AGG 261
#define JOIN 262
#define OUTPUT 263
#define GETS 264
#define PARTITION 265
#define HASHLEFT 266
#define HASHRIGHT 267
#define HASHONE 268
#define UNION 269
#define FLATTEN 270
#define IDENTIFIER 271
#define STRING 272

#endif
