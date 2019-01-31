// Copyright (c) 2018 Ashley Harris / Maptek Australia Pty Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {
using MemoryModelLogicalTest = PassTest<::testing::Test>;

std::string oldPreamble1 = R"(OpCapability Shader
OpCapability Addresses
OpMemoryModel Physical32 GLSL450)";

std::string newPreamble1 = R"(OpCapability Shader
OpCapability Addresses
OpMemoryModel Logical GLSL450)";

std::string unchangingMiddle1 = R"(
OpEntryPoint Fragment %main "main" %pFragColour
OpExecutionMode %main OriginLowerLeft
OpName %Tvoid "Tvoid"
OpName %Tf_Tvoid "Tf_Tvoid"
OpName %Tf32 "Tf32"
OpName %Ti32u "Ti32u"
OpName %Tarr8Tf32 "Tarr8Tf32"
OpName %TppTarr8Tf32 "TppTarr8Tf32"
OpName %TppTf32 "TppTf32"
OpName %Tvec4 "Tvec4"
OpName %TppTvec4 "TppTvec4"
OpName %TpoTvec4 "TpoTvec4"
OpName %pFragColour "pFragColour"
OpName %Lmain "Lmain"
OpName %main "main"
OpDecorate %Tarr8Tf32 ArrayStride 4
OpDecorate %pFragColour Location 0
%Tvoid = OpTypeVoid
%Tf_Tvoid = OpTypeFunction %Tvoid
%Tf32 = OpTypeFloat 32
%Ti32u = OpTypeInt 32 0
%Ti32u_8 = OpConstant %Ti32u 8
%Tarr8Tf32 = OpTypeArray %Tf32 %Ti32u_8
%TppTarr8Tf32 = OpTypePointer Function %Tarr8Tf32
%TppTf32 = OpTypePointer Function %Tf32
%Ti32u_0 = OpConstant %Ti32u 0
%Ti32u_1 = OpConstant %Ti32u 1
%Ti32u_2 = OpConstant %Ti32u 2
%Ti32u_3 = OpConstant %Ti32u 3
%Ti32u_4 = OpConstant %Ti32u 4
%Ti32u_5 = OpConstant %Ti32u 5
%Ti32u_6 = OpConstant %Ti32u 6
%Ti32u_7 = OpConstant %Ti32u 7
%Ti32u_16 = OpConstant %Ti32u 16
%Tf32_1 = OpConstant %Tf32 1
%Tf32_0 = OpConstant %Tf32 0
%Tvec4 = OpTypeVector %Tf32 4
%TppTvec4 = OpTypePointer Function %Tvec4
%TpoTvec4 = OpTypePointer Output %Tvec4
%pFragColour = OpVariable %TpoTvec4 Output
%main = OpFunction %Tvoid None %Tf_Tvoid
%Lmain = OpLabel)";

std::string unchangingMiddle2 = R"(
OpEntryPoint Fragment %1 "main" %2
OpExecutionMode %1 OriginLowerLeft
%void = OpTypeVoid
%float = OpTypeFloat 32
%v3float = OpTypeVector %float 3
%v4float = OpTypeVector %float 4
%_ptr_Function_v3float = OpTypePointer Function %v3float
%_ptr_Function_v4float = OpTypePointer Function %v4float
%9 = OpTypeFunction %void
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%15 = OpConstantComposite %v3float %float_1 %float_0 %float_0
%_ptr_Output_v4float = OpTypePointer Output %v4float
%2 = OpVariable %_ptr_Output_v4float Output
%1 = OpFunction %void None %9
%17 = OpLabel)";

TEST_F(MemoryModelLogicalTest, WriteConstFloatsToArrayBitCastToVectorLoad) {
  // Handle an OpBitCast on a ptr, array to vector, with a constant propagation.

  std::string oldEnding = R"(
%10 = OpVariable %TppTarr8Tf32 Function
%12 = OpInBoundsAccessChain %TppTf32 %10 %Ti32u_0
%14 = OpInBoundsAccessChain %TppTf32 %10 %Ti32u_1
%16 = OpInBoundsAccessChain %TppTf32 %10 %Ti32u_2
%18 = OpInBoundsAccessChain %TppTf32 %10 %Ti32u_3
OpStore %12 %Tf32_0
OpStore %14 %Tf32_1
OpStore %16 %Tf32_1
OpStore %18 %Tf32_1
%24 = OpBitcast %TppTvec4 %10
%25 = OpLoad %Tvec4 %24
OpStore %pFragColour %25
OpReturn
OpFunctionEnd)";

  std::string newEnding = R"(
%26 = OpCompositeConstruct %Tvec4 %Tf32_0 %Tf32_1 %Tf32_1 %Tf32_1
OpStore %pFragColour %26
OpReturn
OpFunctionEnd
)";

  AddPass<MemoryModelToLogical>();
  AddPass<CompactIdsPass>();

  RunAndCheck(oldPreamble1 + unchangingMiddle1 + oldEnding,
              newPreamble1 + unchangingMiddle1 + newEnding);
}

TEST_F(MemoryModelLogicalTest, BitCastWithConstPointerArithmatic) {
  std::string oldEnding = R"(
%10 = OpVariable %TppTarr8Tf32 Function

%11 = OpCompositeConstruct %Tvec4 %Tf32_1 %Tf32_1 %Tf32_0 %Tf32_1
%12 = OpCompositeConstruct %Tvec4 %Tf32_1 %Tf32_0 %Tf32_1 %Tf32_1

; Store one vec4 into the first 4 elements in a single write
%13 = OpBitcast %Ti32u %10
%14 = OpBitcast %TppTvec4 %10
OpStore %14 %11

; Calculate the address of the last 4 elements, and store 4 floats there.
%15 = OpIAdd %Ti32u %13 %Ti32u_16
%16 = OpBitcast %TppTvec4 %15
OpStore %16 %12

; Calculate an address for the 3rd, 4th, 5th and 6th elements as a single vector.
%17 = OpIAdd %Ti32u %13 %Ti32u_8
%24 = OpBitcast %TppTvec4 %17

; Load from the calculated pointer
%25 = OpLoad %Tvec4 %24
OpStore %pFragColour %25
OpReturn
OpFunctionEnd)";

  // We should be able to track it all the way back to constants ideally...
  std::string newEndingIdeal = R"(
%21 = OpCompositeConstruct %Tvec4 %Tf32_0 %Tf32_1 %Tf32_0 %Tf32_1
OpStore %pFragColour %21
OpReturn
OpFunctionEnd
)";

  // But currently we just output the following, valid, but needing another
  // pass, code:
  std::string newEndingValidButNotOptimal = R"(
%26 = OpCompositeConstruct %Tvec4 %Tf32_1 %Tf32_1 %Tf32_0 %Tf32_1
%27 = OpCompositeConstruct %Tvec4 %Tf32_1 %Tf32_0 %Tf32_1 %Tf32_1
%28 = OpCompositeExtract %Tf32 %27 1
%29 = OpCompositeExtract %Tf32 %27 0
%30 = OpCompositeExtract %Tf32 %26 3
%31 = OpCompositeExtract %Tf32 %26 2
%32 = OpCompositeConstruct %Tvec4 %31 %30 %29 %28
OpStore %pFragColour %32
OpReturn
OpFunctionEnd
)";

  AddPass<MemoryModelToLogical>();
  AddPass<CompactIdsPass>();

  RunAndCheck(oldPreamble1 + unchangingMiddle1 + oldEnding,
              newPreamble1 + unchangingMiddle1 + newEndingValidButNotOptimal);
}

TEST_F(MemoryModelLogicalTest, BitCastObservedClangOobRead) {
  // Clang has been observed to output code like this. This is a substantial
  // headache, but if the alignment guarentees of Tvec3 and Tvec4 are both 16,
  // then this code is valid, I guess.

  std::string oldEnding = R"(
; Put a Tvec3 constant into memory
%18 = OpVariable %_ptr_Function_v3float Function %15

; Bitcast it to a Tvec4 and then... load it, reading garbage into the w position.
%19 = OpBitcast %_ptr_Function_v4float %18
%20 = OpLoad %v4float %19

; But without using the undefined w value, insert 1.0 into w
%21 = OpCompositeInsert %v4float %float_1 %20 3

; And store it in output
OpStore %2 %21

OpReturn
OpFunctionEnd)";

  std::string newEnding = R"(
%18 = OpUndef %float
%19 = OpCompositeExtract %float %15 2
%20 = OpCompositeExtract %float %15 1
%21 = OpCompositeExtract %float %15 0
%22 = OpCompositeConstruct %v4float %21 %20 %19 %18
%23 = OpCompositeInsert %v4float %float_1 %22 3
OpStore %2 %23
OpReturn
OpFunctionEnd
)";

  AddPass<MemoryModelToLogical>();
  AddPass<CompactIdsPass>();

  RunAndCheck(oldPreamble1 + unchangingMiddle2 + oldEnding,
              newPreamble1 + unchangingMiddle2 + newEnding);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
