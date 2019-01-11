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

#ifndef LIBSPIRV_OPT_MEMORY_MODEL_TO_LOGICAL_H_
#define LIBSPIRV_OPT_MEMORY_MODEL_TO_LOGICAL_H_

#include "pass.h"

#include <functional>
#include <tuple>

namespace spvtools {
namespace opt {
// This class helps generate shaders from a complex subset of
// SPIR-V, notable those with OpMemoryModel declaring an AddressingModel not
// equal to Logical. Hand written SPIR-V, or SPIR-V compiled from GLSL, is
// unlikely to use these features, however SPIR-V generated via LLVM from
// front-ends like C++ are going to be heavy in the use of pointer arithmetic,
// reinterpret_cast (OpBitCast on a pointer), and other language features that
// are not expressible in GLSL, or are even loadable into vulkan shader
// pipeline.
//
// This class analyses all memory usage of a shader, to help turn complex code
// coming out of a C++-style optimisation pass into simpler SPIR-V
//
// For example:
//
//   OpMemoryModel(Physical64,...)
//   ...
//   %1 = OpVariable<Array<vec3, 2 members, 12 byte stride>>()
//   %2 = OpConstant(sizeof(Tvec2))
//   %3 = (... non constant)
//   %4 = (... non constant)
//   %5 = OpAccessChain(%1, 0)
//   OpStore(%5, %3)
//   %6 = OpAccessChain(%1, 1)
//   OpStore(%6, %4)
//   %7 = OpBitCast<Ptr<vec4>>(%5)
//   %8 = OpBitCast<uint64_t>(%5)
//   %9 = OpAdd(%8, %2)
//   %10 = OpBitCast<Ptr<vec4>>(%9)
//   %11 = OpLoad(%7)
//   %12 = OpLoad(%10)
//   (Usage of %11 and %12)
//
// Where this code hand-written it'd be unlikely to be expressed this way,
// however a compiler may write similar code for a general purpose target (eg
// CPU / GPU / LLVM), and then a toolchain may convert it into valid SPIRV. We
// need to decode this such that there's no pointers, including tracking changes
// to the integer %8, and, ideally, optimising out the %1 variable.
//
// In this example, by placing the %1 array in memory, and writing the %3 and %4
// values to it, tracking the pointer maths, and then simulating the loads, we
// detect that %11 and %12 are expressible as OpShuffle on vectors. Ideally when
// fed into SPIRV-Cross we should get:
//
//   vec3 _3 = ...;
//   vec3 _4 = ...;
//   _11 = vec4(_3, _4.x)
//   _12 = vec4(_3.z, _4)
//
//
class MemoryModelToLogical : public Pass {
 public:
  const char* name() const override { return "memory-model-to-logical"; }
  Status Process() override;

 private:
  // We can run for 32-bit and 64-bit pointer systems.
  uint32_t ptr_width = sizeof(uint32_t);

  // Represents an abstract data concept.
  struct Data {
    virtual ~Data() {}

    enum ResolutionValueComplexity {
      // Value can be anything we feel like.
      Undefined,

      // Some other temporary from somewhere else in the code, unchanged.
      ExistingId,

      // A (new) constant value (or chain of constants).
      Constant,

      // A new SPIR-V sequence (1 or more opcodes) returning an expression, but
      // does not hit the control flow graph.
      SPIRVExpression,

      // Adds new SpirV and new phi nodes.
      SPIRVExpressionAndPhi,

      // Modifies the control flow. So adds branching, looping, etc. Used when
      // writing to a pointer that's the result of pointer arithmetic, or when
      // implementing unknown sized OpMemCopySized instrucitons.
      NewControlFlowGraph,

      // User needs to provide a hint. Eg max object count, integer range,
      // pointer boundaries and alignment, etc.
      NotPossibleWithoutHints,

      // We can't resolve this value. At all. Ever.
      NotPossible,
    };

    // All the data we depend on.
    virtual std::vector<std::shared_ptr<Data>> Dependancies() const;

    // Information about how resolved this data is - see above.
    virtual ResolutionValueComplexity Resolved() const;

    // The id of existing spir-v which expresses this data.
    virtual bool HasExistingId() const { return false; }
    virtual std::vector<uint32_t> ExistingIds() const { return {}; }

    // The address towards which this pointer points.
    virtual bool HasPtr() const { return false; }
    virtual std::vector<uint64_t> PtrValues() const { return {}; }

    // The value stored here, as an integer.
    virtual bool HasInt() const { return false; }
    virtual std::vector<int64_t> IntValues() const { return {}; }

    // The value stored here, as a float.
    virtual bool HasFloat() const { return false; }
    virtual std::vector<double> FloatValues() const { return {}; }

    enum ResolutionCfgComplexity {
      // Unknown
      NoResolution,

      // We know the value, it never changes.
      OneValue,

      // We know all the values, but it'll be different dependant on a branch
      // taken
      MultipleBranchingValues,

      // We know all the values, but it'll be different dependant on how many
      // times we did a loop.
      MultipleLoopingValues,

      // We know all the values, but it depends on factors beyond our control.
      MultipleExternalValues,

      // We know all the values, but it's loaded and stored many times within a
      // single block
      MultipleOverwrittenValues,

      // We know all the values, but can't give a simple answer as to which
      // value occurs when.
      MultipleValues
    };

    virtual ResolutionCfgComplexity ResolvedCfgComplexity() const;

    // Returns either:
    // - an instruction list, with the last instruction returning the result.
    // - A fragment of a control flow graph. The first block with setup and a
    //   branch at the end. All other blocks begining with a label, the last
    //   block being reached through all control flows, and the last statement
    //   ending with the result.
    virtual std::list<InstructionList> ConvertToSpirv(
        IRContext* Context, MemoryModelToLogical* Converter);
  };

  using DataPtrT = std::shared_ptr<Data>;

  // Represents a joining of data together. (eg vec2(const1, data2))
  // Typically the result of reading a composite type from another.
  struct DataComposition : Data {
    std::vector<DataPtrT> values;

    analysis::Type* output_type = nullptr;

    std::vector<std::shared_ptr<Data>> Dependancies() const override {
      return values;
    }

    ResolutionValueComplexity Resolved() const override {
      return std::max(Data::Resolved(), SPIRVExpression);
    }

    std::list<InstructionList> ConvertToSpirv(
        IRContext* Context, MemoryModelToLogical* Converter) override;
  };

  // Represents a value-preserving-ish cast. eg ivec2(uvec2(0, 1))
  struct DataValueCast : Data {
    DataPtrT value;
    analysis::Type* from_type = nullptr;
    analysis::Type* to_type = nullptr;

    std::vector<std::shared_ptr<Data>> Dependancies() const override {
      return {value};
    }

    bool HasFloat() const override { return to_type->AsFloat(); }
    bool HasInt() const { return to_type->AsInteger(); }

    std::vector<double> FloatValues() const;
    std::vector<int64_t> IntValues() const;
  };

  // Represents a change in memory values branch based on paths taken through
  // the control flow graph. So either:
  // - The value we wrote depends on which path.
  // - Where we wrote it depends on which path we took.
  struct DataPhiBranch : Data {
    // NYI
  };

  // Represents an unconditional change in memory value, or a change within
  // a block when multiply read and written without control flow branching.
  struct DataReplaced : Data {
    // NYI
  };

  // Represents a branch based on something that isn't known at compile time and
  // isn't from the control flow graph. eg "float a = *(floatPtr + rand() % 8);"
  struct DataDynamicBranch : Data {
    // NYI
  };

  // A value that exists elsewhere in the SPIR-V
  struct ExistingValue : Data {
    uint32_t id;

    bool HasExistingId() const { return true; }
    std::vector<uint32_t> ExistingIds() const { return {id}; }
  };

  // A specialisation constant that exists elsewhere in the SPIR-V
  struct ExistingSpecialisationConstant : ExistingValue {
    // NYI
  };

  // A constant value
  struct DataConstant : Data {
    std::unique_ptr<analysis::Constant> value;

    // The value stored here, as an integer.
    bool HasInt() const { return value->AsIntConstant(); }
    std::vector<int64_t> IntValues() const;

    // The value stored here, as a float.
    bool HasFloat() const { return value->AsFloatConstant(); }
    std::vector<double> FloatValues() const;

    std::list<InstructionList> ConvertToSpirv(
        IRContext* Context, MemoryModelToLogical* Converter) override;
  };

  // Some input memory we don't control, ie uniforms or vertex attributes coming
  // in.
  struct ExternalData : Data {
    uint64_t address;
  };

  // Step into some other data at a known offset.
  struct DataConstAccessChain : Data {
    DataPtrT value;
    std::vector<uint32_t> steps;
    uint64_t offset = 0;

    std::vector<std::shared_ptr<Data>> Dependancies() const override;

    ResolutionValueComplexity Resolved() const override;

    bool HasPtr() const;
    std::vector<uint64_t> PtrValues() const;
  };

  // Step into some other data using a dynamic index.
  struct DataDynamicAccessChainLink : Data {
    DataPtrT value;
    DataPtrT step;
    uint64_t stride = 1;

    std::vector<std::shared_ptr<Data>> Dependancies() const override;

    ResolutionValueComplexity Resolved() const override;

    bool HasPtr() const;
    std::vector<uint64_t> PtrValues() const;
  };

  // Data grouped together at the bit level. Eg LLVM may move 2 adjacent floats
  // in one instruction by bit casting from and to address to a uint64_t
  // ptr, loading it, and then storing it.
  struct DataBitJoin : Data {
    std::vector<DataPtrT> values;
  };

  // Takes a chunk of data at a bitwise level. Used when reads / writes are
  // misaligned.
  struct DataBitSplit : Data {
    DataPtrT value;
    uint32_t bit_offset;
    uint32_t bit_count;
  };

  // Create a pointer to a tree node.
  struct DataPointerTo : Data {
    uint64_t address;

    bool HasPtr() const { return true; }
    std::vector<uint64_t> PtrValues() const { return {address}; }
  };

  // Use two values to make a new one
  struct DataBinaryOp : Data {
    DataPtrT left;
    DataPtrT right;
    SpvOp opcode;

    bool HasPtr() const;
    std::vector<uint64_t> PtrValues() const;
  };

  // Modify the data in some way
  struct DataUnaryOp : Data {
    DataPtrT value;
    SpvOp opcode;
  };

  // Load memory from an address
  struct DataDereference : Data {
    DataPtrT address_value;
    std::map<uint64_t, DataPtrT> address_to_data;
  };

  // Represents our memory tree.
  struct TreeNode {
    // Where the memory is in our virtual layout.
    uint64_t address;
    uint64_t size_in_bytes = 0;

    // The chain required to get to this node from the parent.
    // May be more than one as single-element structs or size[1] arrays can be
    // nested.
    std::vector<uint32_t> local_access_chain;

    // If known, the variable_id holding this allocation.
    uint32_t owning_variable_id = 0;

    // Whether we have a parent (eg if we're a element in an array).
    bool has_parent = false;

    // Whether we have a child (eg if we're a struct with members).
    bool has_child = false;

    // Our type.
    analysis::Type* type;

    SpvStorageClass storage_class = SpvStorageClassGeneric;

    // Our child entries. (Eg subdivided memory)
    std::vector<TreeNode> children;

    // Value written to the memory
    std::shared_ptr<Data> value;
  };

  // Information about a pointer.
  struct PointerDetails {
    // The type pointed to by the pointer.
    analysis::Type* type;

    // If an exact address is not known, we track the morphing of the pointer
    // back from it's origin - this may of been loaded from memory, or have
    // gone through maths.
    std::shared_ptr<Data> address;

    // Whether the pointer can hold an unlimited number of addresses.
    // For example, c++ iterators may compile to pointer maths, which
    // results in an advancing pointer in a loop. If the loop is unknown bounds,
    // we're going to have trouble converting this to an array lookup.
    //
    // Specifying a bound may allow this to be represented as an array.
    bool unbounded_address_count = false;

    // Whether we're observed to hold a value derived from a pointer, but we
    // also observed to hold a value not derived from a pointer.
    bool holds_both_pointer_and_non_pointer = false;

    // Whether the address could ever be loaded from memory.
    bool address_ever_loaded_from_memory = false;
  };

  // Returns a value (eg the code for an OpLoad) that was previously dependant
  // on complex memory details.
  const Data* value_generated(uint32_t Id) const {
    auto it = id_data.find(Id);
    if (it != id_data.end()) return it->second.get();
    return nullptr;
  }

  // Tracks each variable which holds a pointer, or an integer casted from a
  // pointer.
  std::unordered_map<uint32_t, PointerDetails> stack_pointer_details;

  // As above, but tracks pointers stored in memory.
  std::unordered_map<uint64_t, PointerDetails> heap_pointer_details;

  // Returns every address which this pointer may be pointing at.
  // If the bounds are infinite (or practically infinite for the purpose of
  // generating branching glsl code), then an empty vector is returned.
  std::vector<uint64_t> all_potential_stack_pointer_destinations(
      uint32_t Id) const;

  // As above, but if no hints have been given regarding alignment and values,
  // will walk the memory tree to see what sizes we might be wanting to work
  // with based on what copy sizes would avoid undefined behaviour.
  std::vector<int64_t> all_potential_sensible_mem_copy_sizes(
      uint32_t SizeId, uint64_t SourcePtrAddress, uint64_t DestPtrAddress);

  std::vector<int64_t> all_potential_int_values(uint32_t Id) const;

  // Tracks all memory allocated or otherwise referenced by the shader.
  std::vector<TreeNode> memory_tree;

  // Returns the size of a type on this platform (as we can have pointer members
  // of structs this is memory model specific).
  uint32_t size_of(const analysis::Type& Type) const;
  std::vector<uint32_t> child_types(const analysis::Type& Type) const;
  bool type_holds_pointer(const analysis::Type& Type) const;

  // Populate the entire heap memory tree's children with details. This includes
  // identifying pointers in the heap.
  void recurse_populate_memory_tree(TreeNode* Node);

  // See if we can get a perfect, uncasted, aligned, read of memory.
  TreeNode* try_aligned_uncasted_memory_access(
      const analysis::Type& DesiredType, uint64_t Address, uint32_t Size);

  // See if we can get a read of memory but need casts.
  std::vector<TreeNode*> try_aligned_access_memory(
      const analysis::Type& DesiredType, uint64_t Address, uint32_t Size);

  // Try to find the tree block best containing memory. To implement a read /
  // write to this memory we may need to write casts, shifts, float assembly /
  // disassembly / and other horrible things.
  TreeNode* find_memory_location(uint64_t Address, uint32_t Size);

  // Find the "root" allocation, ie the "OpVariable"
  TreeNode* find_memory_root_allocation(uint64_t Address);

  // Find the "leaf" allocation, ie the lowest member.
  TreeNode* find_memory_leaf_allocation(uint64_t Address);

  // Called multiple times, returns true if the potential pointer set grew from
  // detecting pointers being casted to ints, those ints modified, and then
  // casted back to pointers. Also detects when a non-pointer (and non constexpr
  // 0) value is stored in an integer previously identified as also holding a
  // pointer.
  bool find_and_track_pointers(const BasicBlock* Block);

  // Called once per block once all pointers (and int's that are really
  // pointers) are known, different behaviour caused by Phi nodes changing
  // values is processed here too. Returns whether anything was learnt.
  bool process(BasicBlock* Block, uint32_t Precessor);

  // Decodes an access chain (that's all constants)
  std::pair<uint32_t, uint32_t> access_chain_to_byte_offset_and_size(
      const analysis::Type& Type, uint32_t ThisStep,
      Instruction::const_iterator NextStep,
      Instruction::const_iterator EndOfSteps);

  // DataRef address_last_write_ref(TreeNode *Address, uint32_t CurrentBlock,
  // uint32_t Predesessor) const; DataRef temporary_to_ref(uint32_t Id);

  // Simulates operations - returns true if anything new was learnt.
  bool simulate_memcopy(uint64_t SourceAddress, uint64_t DestinationAddress,
                        uint32_t SizeInBytes);
  bool simulate_store(uint32_t SourceId, uint64_t DestinationAddress,
                      uint32_t SizeInBytes);

  std::unordered_map<uint32_t, std::shared_ptr<Data>> id_data;
  std::unordered_set<uint32_t> variables_to_cull;

  void splice_before(InstructionList&& Instructions, Instruction* Destination);

  analysis::Integer* GetUint32Type();
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_MEMORY_MODEL_TO_LOGICAL_H_
