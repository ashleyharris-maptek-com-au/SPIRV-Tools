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

#include "memory_model_to_logical.h"

#include "source/opt/ir_context.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
std::vector<uint64_t>
MemoryModelToLogical::all_potential_stack_pointer_destinations(
    uint32_t Id) const {
  //:TODO: if code called set_hint_ptr_bounds, set_hint_ptr_address, etc, use
  // that.

  auto it = stack_pointer_details.find(Id);
  if (it == stack_pointer_details.end()) {
    assert(0 && "Not a pointer?");
  }

  std::vector<uint64_t> outputs;

  assert(it->second.address->HasPtr());

  outputs = it->second.address->PtrValues();

  return outputs;
}

std::vector<int64_t>
MemoryModelToLogical::all_potential_sensible_mem_copy_sizes(
    uint32_t SizeId, uint64_t SourcePtrAddress, uint64_t DestPtrAddress) {
  return {};
}

inline std::vector<int64_t> MemoryModelToLogical::all_potential_int_values(
    uint32_t Id) const {
  // So - it'd be nice if the length is a constant.
  auto* constant = context()->get_constant_mgr()->FindDeclaredConstant(Id);

  if (constant) {
    auto intConstant = constant->AsIntConstant();
    if (intConstant->words().size() == 1) {
      if (intConstant->type()->AsInteger()->IsSigned()) {
        return {intConstant->GetS32()};
      }
      return {intConstant->GetU32()};
    }

    if (intConstant->type()->AsInteger()->IsSigned()) {
      return {intConstant->GetS64()};
    }
    return {int64_t(intConstant->GetU64())};
  }

  // Not a constant - try to read it from our id tracking
  auto it = id_data.find(Id);
  if (it != id_data.end()) {
    return it->second->IntValues();
  }

  // Has the user set a hint - use that.

  // NYI

  // We don't know what the value is.
  return {};
}

uint32_t MemoryModelToLogical::size_of(const analysis::Type& Type) const {
  auto typeAsId = context()->get_type_mgr()->GetId(&Type);

  if (Type.AsPointer()) {
    // Pointers have a fixed size, regardless of what they're pointing to.
    return ptr_width;
  }

  if (Type.AsArray()) {
    auto arrayInfo = Type.AsArray();

    uint32_t stride = 0;

    context()->get_decoration_mgr()->ForEachDecoration(
        typeAsId, SpvDecorationArrayStride,
        [&](const Instruction& I) { stride = I.GetSingleWordOperand(2); });

    assert(stride);

    auto arraySizeId = arrayInfo->LengthId();
    auto potentialLengths = all_potential_int_values(arraySizeId);

    if (potentialLengths.empty()) {
      // Need to ask the user for a hint, error out, and retry.
      assert(0);
    }

    auto lengthElement =
        std::max_element(potentialLengths.begin(), potentialLengths.end());
    return uint32_t(*lengthElement) * stride;
  }

  if (Type.AsStruct()) {
    auto asStruct = Type.AsStruct();

    uint32_t offsetLast = 0;

    auto& elementDecorations = asStruct->element_decorations();

    auto elementEntry =
        elementDecorations.find(uint32_t(asStruct->element_types().size() - 1));

    for (auto& deco : elementEntry->second) {
      if (deco.front() == SpvDecorationOffset) {
        offsetLast = deco.back();
      }
    }

    auto sizeofLast = size_of(*asStruct->element_types().back());

    // We don't try to guess padding - as padding is added by the array /
    // struct holding this type. So sizeof(pair(int, bool)) == 5.
    return offsetLast + sizeofLast;
  }

  if (auto vector = Type.AsVector()) {
    return vector->element_count() * size_of(*vector->element_type());
  }

  if (auto matrix = Type.AsMatrix()) {
    return matrix->element_count() * size_of(*matrix->element_type());
  }

  if (auto boo = Type.AsBool()) {
    // Bools can't be stored - so this is an int.
    return 4;
  }

  if (auto floa = Type.AsFloat()) {
    return floa->width() / 8;
  }

  if (auto in = Type.AsInteger()) {
    return in->width() / 8;
  }

  assert(0);
  return 0;
}

std::vector<uint32_t> MemoryModelToLogical::child_types(
    const analysis::Type& Type) const {
  auto typeAsId = context()->get_type_mgr()->GetId(&Type);

  if (Type.AsArray()) {
    auto arrayInfo = Type.AsArray();

    auto arraySizeId = arrayInfo->LengthId();
    auto potentialLengths = all_potential_int_values(arraySizeId);

    if (potentialLengths.empty()) {
      // Need to ask the user for a hint, error out, and retry.
      assert(0);
    }

    auto elementId =
        context()->get_type_mgr()->GetId(arrayInfo->element_type());

    auto lengthElement =
        std::max_element(potentialLengths.begin(), potentialLengths.end());
    return std::vector<uint32_t>(uint32_t(*lengthElement), elementId);
  }

  if (Type.AsStruct()) {
    auto asStruct = Type.AsStruct();
    std::vector<uint32_t> members;

    for (auto& member : asStruct->element_types()) {
      members.push_back(context()->get_type_mgr()->GetId(member));
    }
    return members;
  }

  if (auto vector = Type.AsVector()) {
    auto elementId = context()->get_type_mgr()->GetId(vector->element_type());
    return std::vector<uint32_t>(vector->element_count(), elementId);
  }

  if (auto matrix = Type.AsMatrix()) {
    auto elementId = context()->get_type_mgr()->GetId(matrix->element_type());
    return std::vector<uint32_t>(matrix->element_count(), elementId);
  }

  assert(0);
  return {};
}

bool MemoryModelToLogical::type_holds_pointer(
    const analysis::Type& Type) const {
  // NYI
  return false;
}

void MemoryModelToLogical::recurse_populate_memory_tree(TreeNode* Node) {
  if (auto ptr = Node->type->AsPointer()) {
    // This is a pointer stored in heap memory. Add it to our pointer tracking.

    auto& pd = heap_pointer_details[Node->address];
    pd.type = context()->get_type_mgr()->GetRegisteredType(ptr->pointee_type());

    return;
  }

  if (auto struc = Node->type->AsStruct()) {
    Node->has_child = true;

    // We have to iterate through the struct members and place them in memory.
    // Luckily there is the Offset decoration to make this super easy.

    auto& elementDecorations = struc->element_decorations();
    for (auto childIndex = 0u; childIndex < struc->element_types().size();
         childIndex++) {
      TreeNode node;
      node.type = context()->get_type_mgr()->GetRegisteredType(
          struc->element_types()[childIndex]);

      auto elementEntry = elementDecorations.find(childIndex);

      uint32_t offset;

      for (auto& deco : elementEntry->second) {
        if (deco.front() == SpvDecorationOffset) {
          offset = deco.back();
        }
      }

      auto size = size_of(*struc->element_types()[childIndex]);

      node.address = Node->address + offset;
      node.size_in_bytes = size;
      node.has_parent = true;
      node.local_access_chain = Node->local_access_chain;
      node.local_access_chain.push_back(childIndex);

      auto genesis = find_memory_root_allocation(node.address);
      if (genesis->value) {
        // We're not undef - so set our value as a reference to our ancestors
        // value + access chain.
        auto valueChain = std::make_shared<DataConstAccessChain>();
        valueChain->steps = node.local_access_chain;
        valueChain->value = genesis->value;
        valueChain->offset = node.address - genesis->address;
        node.value = valueChain;
      }

      Node->children.push_back(std::move(node));
    }
  } else if (auto array = Node->type->AsArray()) {
    Node->has_child = true;

    TreeNode node;
    node.type =
        context()->get_type_mgr()->GetRegisteredType(array->element_type());

    auto childSize = size_of(*array->element_type());
    node.size_in_bytes = childSize;
    node.has_parent = true;

    uint32_t stride = 0;

    context()->get_decoration_mgr()->ForEachDecoration(
        context()->get_type_mgr()->GetId(array), SpvDecorationArrayStride,
        [&](const Instruction& I) { stride = I.GetSingleWordOperand(2); });

    assert(stride);

    auto arraySizeId = array->LengthId();
    auto potentialLengths = all_potential_int_values(arraySizeId);

    if (potentialLengths.empty()) {
      // Need to ask the user for a hint, error out, and retry.
      assert(0);
    }

    auto lengthElement =
        std::max_element(potentialLengths.begin(), potentialLengths.end());

    auto arrayLength = *lengthElement;

    node.local_access_chain = Node->local_access_chain;
    node.local_access_chain.push_back(0);

    for (auto childIndex = 0u; childIndex < arrayLength; childIndex++) {
      node.address = Node->address + childIndex * stride;
      node.local_access_chain.back() = childIndex;

      Node->children.push_back(node);
    }
  } else if (auto matrix = Node->type->AsMatrix()) {
    uint32_t stride;

    context()->get_decoration_mgr()->ForEachDecoration(
        context()->get_type_mgr()->GetId(array), SpvDecorationMatrixStride,
        [&](const Instruction& I) { stride = I.GetSingleWordOperand(2); });

    TreeNode node;
    node.local_access_chain = Node->local_access_chain;
    node.local_access_chain.push_back(0);
    node.type =
        context()->get_type_mgr()->GetRegisteredType(matrix->element_type());
    node.has_parent = true;
    node.size_in_bytes = size_of(*matrix->element_type());

    for (auto childIndex = 0u; childIndex < matrix->element_count();
         childIndex++) {
      node.address = Node->address + childIndex * node.size_in_bytes;
      node.local_access_chain.back() = childIndex;

      Node->children.push_back(node);
    }
  } else if (auto vector = Node->type->AsVector()) {
    Node->has_child = true;

    TreeNode node;
    node.local_access_chain = Node->local_access_chain;
    node.local_access_chain.push_back(0);
    node.type =
        context()->get_type_mgr()->GetRegisteredType(vector->element_type());
    node.has_parent = true;
    node.size_in_bytes = size_of(*vector->element_type());

    for (auto childIndex = 0u; childIndex < vector->element_count();
         childIndex++) {
      node.address = Node->address + childIndex * node.size_in_bytes;
      node.local_access_chain.back() = childIndex;

      Node->children.push_back(node);
    }
  } else {
    // Scalar type.
    return;
  }

  if (Node->has_child) {
    for (auto& child : Node->children) {
      recurse_populate_memory_tree(&child);
    }
  }
}

MemoryModelToLogical::TreeNode*
MemoryModelToLogical::try_aligned_uncasted_memory_access(
    const analysis::Type& DesiredType, uint64_t Address, uint32_t Size) {
  auto block = find_memory_location(Address, Size);

  if (block->address == Address && block->size_in_bytes == Size &&
      block->type == &DesiredType) {
    // That was lucky!
    return block;
  }

  return nullptr;
}

std::vector<MemoryModelToLogical::TreeNode*>
MemoryModelToLogical::try_aligned_access_memory(
    const analysis::Type& DesiredType, uint64_t Address, uint32_t Size) {
  auto block = find_memory_location(Address, Size);

  auto sizeOf = size_of(DesiredType);

  if (block->size_in_bytes < Size) {
    assert(0 &&
           "Buffer overrun. We might need to insert undefined values for "
           "certain known "
           "safe cases - eg loading a vec3 aligned to 16 bytes as a vec4 but "
           "only accessing the "
           "first 3 elements.");
  }

  if (block->type == &DesiredType) {
    // We got a match!
    return {block};
  }

  // We need to try to line up the children
  auto offset = 0;
  auto childTypes = child_types(DesiredType);

  if (childTypes.empty()) {
    // We're at the base of the type heirchay - hopefully in memory we're at
    // scalar level too.
    if (block->size_in_bytes == sizeOf) {
      return {block};
    }
    return {};
  }

  std::vector<TreeNode*> output;

  for (auto child : childTypes) {
    auto& childType = *context()->get_type_mgr()->GetType(child);
    auto sizeOfChild = size_of(childType);
    auto result =
        try_aligned_access_memory(childType, Address + offset, sizeOfChild);
    if (result.empty()) return {};
    output.insert(output.end(), result.begin(), result.end());
    offset += sizeOfChild;
  }

  return output;
}

MemoryModelToLogical::TreeNode* MemoryModelToLogical::find_memory_location(
    uint64_t Address, uint32_t Size) {
  std::vector<TreeNode>* branch = &this->memory_tree;
  TreeNode* output = nullptr;

  while (!branch->empty()) {
    bool inAChild = false;
    for (auto& child : *branch) {
      if (Address >= child.address &&
          Address + Size <= child.address + child.size_in_bytes) {
        inAChild = true;
        branch = &child.children;
        output = &child;

        if (branch->empty()) {
          return &child;
        }
        break;
      }
    }

    if (!inAChild) {
      return output;
    }
  }

  return nullptr;
}

MemoryModelToLogical::TreeNode*
MemoryModelToLogical::find_memory_root_allocation(uint64_t Address) {
  for (auto& child : memory_tree) {
    if (Address >= child.address &&
        Address <= child.address + child.size_in_bytes) {
      return &child;
    }
  }
  return nullptr;
}

MemoryModelToLogical::TreeNode*
MemoryModelToLogical::find_memory_leaf_allocation(uint64_t Address) {
  std::vector<TreeNode>* branch = &this->memory_tree;

  while (!branch->empty()) {
    for (auto& child : *branch) {
      if (Address >= child.address &&
          Address <= child.address + child.size_in_bytes) {
        branch = &child.children;
        if (branch->empty()) {
          return &child;
        }
        break;
      }
    }
  }

  return nullptr;
}

bool MemoryModelToLogical::find_and_track_pointers(const BasicBlock* Block) {
  bool learntSomething = false;
  std::vector<uint32_t> pointersIn;
  std::vector<uint32_t> nonPointerIdsIn;
  std::vector<uint32_t> litteralsIn;

  for (auto& instruction : *Block) {
    pointersIn.clear();
    nonPointerIdsIn.clear();
    litteralsIn.clear();

    for (auto& arg : instruction) {
      if (arg.type == spv_operand_type_t::SPV_OPERAND_TYPE_ID) {
        auto it = this->stack_pointer_details.find(arg.words[0]);
        if (it != this->stack_pointer_details.end()) {
          // We passed a pointer into this instruction.
          pointersIn.push_back(arg.words[0]);
        } else {
          nonPointerIdsIn.push_back(arg.words[0]);
        }
      } else {
        litteralsIn.push_back(arg.words[0]);
      }
    }

    // We can ignore if it takes no pointers or pointer-like-ints.
    if (pointersIn.empty()) continue;

    auto morphPointerNoOp = [&]() {
      if (instruction.HasResultId() && pointersIn.size() == 1) {
        auto returnTypeNo = instruction.GetSingleWordOperand(0);
        auto returnidNo = instruction.GetSingleWordOperand(1);
        auto returnType = context()->get_type_mgr()->GetType(returnTypeNo);

        auto& ptrInfo = stack_pointer_details[returnidNo];
        auto& inputInfo = stack_pointer_details[pointersIn.front()];

        if (!ptrInfo.address.get()) {
          // We've learnt something here!
          ptrInfo = stack_pointer_details[pointersIn.front()];
          learntSomething = true;
        } else if (ptrInfo.address.get() ==
                   stack_pointer_details[pointersIn.front()].address.get()) {
          // We already knew this.
        } else {
          assert(0 && "NYI");
        }
      } else {
        assert(0 &&
               "Instruction took a pointer and didn't output anything, or took "
               "two pointers.");
      }
    };

    auto morphPointerAddConstant = [&](uint32_t offset) {
      if (instruction.HasResultId() && pointersIn.size() == 1) {
        auto returnTypeNo = instruction.GetSingleWordOperand(0);
        auto returnidNo = instruction.GetSingleWordOperand(1);
        auto returnType = context()->get_type_mgr()->GetType(returnTypeNo);

        auto& ptrInfo = stack_pointer_details[returnidNo];
        auto& inputInfo = stack_pointer_details[pointersIn.front()];

        if (ptrInfo.address.get() == nullptr) {
          // We've learnt something here!
          auto offsetData = std::make_shared<DataBinaryOp>();
          offsetData->opcode = SpvOpIAdd;
          offsetData->left = stack_pointer_details[pointersIn.front()].address;

          auto constantData = std::make_shared<DataConstant>();

          auto uint32t = GetUint32Type();

          auto intConstant = std::make_unique<analysis::IntConstant>(
              uint32t, std::vector<uint32_t>{offset});

          constantData->value = std::move(intConstant);

          offsetData->right = constantData;

          if (offset == 0) {
            ptrInfo.address = offsetData->left;
          } else {
            ptrInfo.address = offsetData;
          }

          learntSomething = true;
        } else {
          // We've already analysed this pointer.
        }
      } else {
        assert(0 &&
               "Instruction took a pointer and didn't output anything, or took "
               "two pointers.");
      }
    };

    switch (instruction.opcode()) {
      case SpvOpStore: {
        // OpStore with a single pointer arg can be ignored most of the time
        // (for the purposes of pointer tracking). If it's writing a pointer to
        // memory, or writing to memory that contains a pointer, that's a more
        // complicated case.
        // if (type_holds_pointer(context()->get_type_mgr()->GetType(
        //        type_of_temporary_id(nonPointerIdsIn.front()))))
        //{
        //            assert(0 &&"NYI - pointer in heap tracking");
        //}
        break;
      }
      case SpvOpLoad: {
        // OpLoad on a pointer doesn't (often) return a pointer.
        // if (type_holds_pointer(context()->get_type_mgr()->GetType(ops[0])))
        //{
        //            assert(0 &&"NYI - pointer in heap tracking");
        //}
        break;
      }
      case SpvOpBitcast: {
        // OpBitcast just changes the type of the pointer, or converts it to or
        // from an integer. The validation rules insist that they must have the
        // same number of bits, so it's just a no-op.
        morphPointerNoOp();
        break;
      }
      case SpvOpConvertPtrToU: {
        auto returnTypeNo = instruction.type_id();
        auto typeInfo = context()->get_type_mgr()->GetType(returnTypeNo);

        if (typeInfo->AsInteger()->width() == ptr_width * 8) {
          // Direct pointer cast to int - no op, and bless the integer as a
          // pointer.
          morphPointerNoOp();
        } else {
          // This is technically valid SPIR-V, but it just seems so dubious, and
          // unsafe.
          //
          // (I haven't seen this in the wild.)
          assert(0 &&
                 "Truncation of memory address to a smaller integer type.");
        }
        break;
      }

      case SpvOpConvertUToPtr:
      case SpvOpGenericCastToPtr:
      case SpvOpGenericCastToPtrExplicit:
      case SpvOpPtrCastToGeneric: {
        // These pointer casts are always 'no-ops' - just forward the pointer.
        morphPointerNoOp();
        break;
      }

      case SpvOpUConvert:
      case SpvOpSConvert: {
        auto returnTypeNo = instruction.type_id();
        auto typeInfo = context()->get_type_mgr()->GetType(returnTypeNo);
        if (typeInfo->AsInteger()->width() >= ptr_width * 8) {
          // Casting up a pointer to a larger type... weird, but legal and
          // should be workable.
          morphPointerNoOp();
        } else {
          // This is technically valid SPIR-V, but it just seems so dubious, and
          // unsafe. Maybe we're using uint8_t(size_t(this) >> 8) as a random
          // number generator?
          assert(0 && "truncation of pointer.");
        }

        break;
      }
      case SpvOpFConvert:
      case SpvOpConvertUToF:
      case SpvOpConvertSToF: {
        assert(0 && "Converting a pointer to a floating point?");
      }

      case SpvOpIAdd: {
        // :TODO: find the non-pointer, and track it back to a constant if
        // possible, else try to derive an expression for it.
        assert(0 && "NYI");
        break;
      }

      case SpvOpISub: {
        // Two cases - either:
        // - diff between two pointers, which is later used for something, eg
        //   convert an anonymous access chain into a pointer diff, and then
        //   apply it to any pointer.
        // - Or moving backwards through an array. ew.
        //
        // :TODO: find the non-pointer, and track it back to a constant if
        // possible, else try to derive an expression for it.
        assert(0 && "NYI");

        break;
      }

      case SpvOpUMod:
      case SpvOpUDiv:
      case SpvOpBitwiseAnd:
      case SpvOpBitwiseOr:
      case SpvOpBitwiseXor:
      case SpvOpShiftLeftLogical:
      case SpvOpShiftRightArithmetic:
      case SpvOpShiftRightLogical: {
        // This may be needed for a compilation of something like a
        // boost::intrusive::set or some other non allocating rbtree with an
        // optimisation which stores the red / black flag within the lsb of the
        // pointer.
        assert(0 && "NYI");
      }

      case SpvOpInBoundsAccessChain:
      case SpvOpAccessChain: {
        // We need to step into memory following the indices to calculate an
        // offset.

        // This is basically the C++ arrow operator. These are the same as
        // OpPtrAccessChain and OpInBoundsPtrAccessChain, but with an extra '0'
        // at the start. :-) (See https://llvm.org/docs/GetElementPtr.html)

        auto* inputTypePtr = context()->get_type_mgr()->GetType(
            context()
                ->get_def_use_mgr()
                ->GetDef(instruction.GetSingleWordOperand(2))
                ->type_id());

        auto* inputType = inputTypePtr->AsPointer()->pointee_type();

        auto offsetAndSize = access_chain_to_byte_offset_and_size(
            *inputType, 0, instruction.begin() + 3, instruction.end());

        auto* outputTypePtr =
            context()->get_type_mgr()->GetType(instruction.type_id());
        auto* outputType = outputTypePtr->AsPointer()->pointee_type();

        auto outputSize = size_of(*outputType);

        if (outputSize != offsetAndSize.second) {
          assert(0);
          //           assert(0 &&"Access chain size mismatch?");
        }

        morphPointerAddConstant(offsetAndSize.first);

        break;
      }
      case SpvOpPtrAccessChain:
      case SpvOpInBoundsPtrAccessChain: {
        // We need to step into memory following the indices to calculate an
        // offset. This is basically a C++ square bracket following a pointer.
        // (See https://llvm.org/docs/GetElementPtr.html)

        auto* inputTypePtr = context()->get_type_mgr()->GetType(
            context()
                ->get_def_use_mgr()
                ->GetDef(instruction.GetSingleWordOperand(2))
                ->type_id());

        auto* inputType = inputTypePtr->AsPointer()->pointee_type();

        auto offsetAndSize = access_chain_to_byte_offset_and_size(
            *inputType, instruction.GetSingleWordInOperand(2),
            instruction.begin() + 2, instruction.end());

        auto* outputTypePtr =
            context()->get_type_mgr()->GetType(instruction.type_id());
        auto* outputType = outputTypePtr->AsPointer()->pointee_type();

        auto outputSize = size_of(*outputType);

        if (outputSize != offsetAndSize.second) {
          assert(0);
          //           assert(0 &&"Access chain size mismatch?");
        }

        morphPointerAddConstant(offsetAndSize.first);

        break;
      }
      default:
        assert(0);
        //           assert(0 &&"Pointer passed into a not yet implemented
        // instruction");
    }
  }

  return learntSomething;
}

bool MemoryModelToLogical::process(BasicBlock* Block, uint32_t Precessor) {
  auto defMgr = context()->get_def_use_mgr();
  auto typeMgr = context()->get_type_mgr();

  auto type_of_temporary_id = [&](uint32_t A) {
    auto typeId = defMgr->GetDef(A)->type_id();
    return typeMgr->GetType(typeId);
  };

  Block->ForEachInst([&](Instruction* I) {
    switch (I->opcode()) {
      case SpvOpCopyMemory: {
        auto targetPtrId = I->GetSingleWordInOperand(0);
        auto sourcePtrId = I->GetSingleWordInOperand(1);

        auto targetPointedType =
            type_of_temporary_id(targetPtrId)->AsPointer()->pointee_type();
        auto sourcePointedType =
            type_of_temporary_id(sourcePtrId)->AsPointer()->pointee_type();
        auto sizeOfPointed1 = size_of(*sourcePointedType);
        auto sizeOfPointed2 = size_of(*targetPointedType);

        if (sizeOfPointed1 == 0 || sizeOfPointed1 != sizeOfPointed2) {
          assert(0 &&
                 "OpCopyMemory members must be same type, with valid size");
        }

        auto targetPtrPotentials =
            all_potential_stack_pointer_destinations(targetPtrId);
        auto sourcePtrPotentials =
            all_potential_stack_pointer_destinations(sourcePtrId);

        if (targetPtrPotentials.size() == 1 &&
            sourcePtrPotentials.size() == 1) {
          // Trivial case - 2 pointers which can only ever hold static values.
          simulate_memcopy(sourcePtrPotentials.front(),
                           targetPtrPotentials.front(), sizeOfPointed1);
        } else {
          assert(0 && "NYI");
        }
        break;
      }
      case SpvOpCopyMemorySized: {
        auto targetPtrId = I->GetSingleWordInOperand(0);
        auto sourcePtrId = I->GetSingleWordInOperand(1);
        auto sizeId = I->GetSingleWordInOperand(2);

        auto targetPointedType =
            type_of_temporary_id(targetPtrId)->AsPointer()->pointee_type();
        auto sourcePointedType =
            type_of_temporary_id(sourcePtrId)->AsPointer()->pointee_type();

        auto targetPtrPotentials =
            all_potential_stack_pointer_destinations(targetPtrId);
        auto sourcePtrPotentials =
            all_potential_stack_pointer_destinations(sourcePtrId);

        if (targetPtrPotentials.size() == 1 &&
            sourcePtrPotentials.size() == 1) {
          // Simpler case - 2 pointers which can only ever hold static values.
          // Now the only problem is the size, which need not be constant.

          auto validSizes = all_potential_int_values(sizeId);

          if (validSizes.empty()) {
            validSizes = all_potential_sensible_mem_copy_sizes(
                sizeId, sourcePtrPotentials.front(),
                targetPtrPotentials.front());
          }

          if (validSizes.empty()) {
            assert(0 &&
                   "Can't decode OpCopyMemorySized, as the number of bytes to "
                   "copy couldn't be deduced. ");
          }

          if (validSizes.size() == 1) {
            // Sweet, constant, case.
            simulate_memcopy(sourcePtrPotentials.front(),
                             targetPtrPotentials.front(),
                             uint32_t(validSizes.front()));
          } else {
            // We have to put conditional writes.
            assert(0 && "NYI");
          }
        } else {
          // We have variable address we could write to.
          assert(0 && "NYI");
        }
        break;
      }
      case SpvOpStore: {
        auto targetPtrId = I->GetSingleWordInOperand(0);
        auto dataId = I->GetSingleWordInOperand(1);

        auto targetPointedType =
            type_of_temporary_id(targetPtrId)->AsPointer()->pointee_type();
        auto sourceType = type_of_temporary_id(dataId);

        if (targetPointedType != sourceType) {
          assert(
              0 &&
              "OpStore pointer type must match data type (missing OpBitCast?)");
        }

        auto targetPtrPotentials =
            all_potential_stack_pointer_destinations(targetPtrId);

        if (targetPtrPotentials.empty()) {
          assert(0 &&
                 "OpStore failure - We have no idea where we're writing (use "
                 "set_ptr_hint... functions)");
        } else if (targetPtrPotentials.size() == 1) {
          // Trivial case - pointer which can point to one place.
          simulate_store(dataId, targetPtrPotentials.front(),
                         size_of(*sourceType));
        } else {
          // We have multiple write locations, lets look at control flow
          // analysis.
          targetPtrPotentials =
              all_potential_stack_pointer_destinations(targetPtrId);

          // We have multiple write locations - this needs an "if (a) b = c;
          // else if (d) e = c; else f = c;" style monstrosity.
          assert(0 && "NYI");
        }
        break;
      }

      case SpvOpLoad: {
        // So now we know all id's that may end up stored in memory that we
        // control / know about. Lets see if we can join this load back to the
        // store(s) that made it (with merging, splitting casting, etc glue).
        // The end result (hopefully) is a merging of (two or more) branches on
        // our expression tree, removing all that nasty pointer stuff.

        auto resultType = I->type_id();
        auto resultId = I->result_id();
        auto ptr = I->GetSingleWordInOperand(0);

        auto pointedType =
            type_of_temporary_id(ptr)->AsPointer()->pointee_type();

        auto resultTypeInfo = context()->get_type_mgr()->GetType(resultType);
        auto sizeOf = size_of(*resultTypeInfo);

        if (pointedType != resultTypeInfo) {
          assert(
              0 &&
              "OpLoad pointer type must match data type (missing OpBitCast?)");
        }

        auto sourcePtrPotentials =
            all_potential_stack_pointer_destinations(ptr);

        if (sourcePtrPotentials.size() == 1) {
          // Simple case - pointer which can point to one place.

          auto block = try_aligned_uncasted_memory_access(
              *resultTypeInfo, sourcePtrPotentials.front(), sizeOf);

          if (block) {
            // Nothing special needed.
            id_data[resultId] = block->value;
          } else {
            auto blocks = try_aligned_access_memory(
                *resultTypeInfo, sourcePtrPotentials.front(), sizeOf);

            if (blocks.empty()) {
              // Read isn't aligned - TODO
              assert(0 && "NYI");
            } else {
              if (blocks.size() == 1) {
                // This is a cast
                auto cast = std::make_shared<DataValueCast>();
                cast->value = block->value;
                cast->from_type = block->type;
                cast->to_type = resultTypeInfo;
                id_data[resultId] = cast;
              } else {
                // This is a recomposition
                auto values = std::make_shared<DataComposition>();
                for (auto b : blocks) {
                  values->values.push_back(b->value);
                }
                values->output_type = resultTypeInfo;
                id_data[resultId] = values;
              }
            }
          }
        } else {
          assert(0 && "NYI - multi pointer load");
        }
        break;
      }
    }
  });
  return {};
}

MemoryModelToLogical::Status MemoryModelToLogical::Process() {
  memory_tree.clear();
  stack_pointer_details.clear();
  heap_pointer_details.clear();

  // Having 'nullptr' be a valid address just feels wrong, so start a few bytes
  // in.
  auto address = 64ull;

  // Create a top level memory tree, placing all known OpVariable data.
  get_module()->ForEachInst([&](const Instruction* I) {
    if (I->opcode() != SpvOpVariable) return;

    auto* ptrType = context()->get_type_mgr()->GetType(I->type_id());
    auto* type = ptrType->AsPointer()->pointee_type();

    // Place the allocation in memory
    TreeNode node;
    node.address = address;
    node.size_in_bytes = size_of(*type);
    node.type = context()->get_type_mgr()->GetRegisteredType(type);
    node.has_parent = false;
    node.owning_variable_id = I->result_id();
    node.storage_class = SpvStorageClass(I->GetSingleWordOperand(2));

    if (node.storage_class == SpvStorageClassUniformConstant ||
        node.storage_class == SpvStorageClassInput ||
        node.storage_class == SpvStorageClassUniform ||
        node.storage_class == SpvStorageClassWorkgroup ||
        node.storage_class == SpvStorageClassCrossWorkgroup ||
        node.storage_class == SpvStorageClassGeneric ||
        node.storage_class == SpvStorageClassPushConstant ||
        node.storage_class == SpvStorageClassAtomicCounter ||
        node.storage_class == SpvStorageClassImage ||
        node.storage_class == SpvStorageClassStorageBuffer) {
      // These values have data set externally, which we must represent in our
      // memory tree.

      auto value = std::make_shared<ExternalData>();
      value->address = address;
      node.value = value;
    }

    if (I->NumOperands() > 3) {
      // We have an initialiser - we should note it.
      if (node.value) {
        // We already have defined a value here - ie it's external memory.
        // This means the initialiser is a default value, eg a uniform with a
        // fallback. We can't know the value of the memory.
      } else {
        auto constantId = I->GetSingleWordOperand(3);

        auto constant =
            context()->get_constant_mgr()->FindDeclaredConstant(constantId);

        if (constant) {
          auto dataConstant = std::make_shared<DataConstant>();
          dataConstant->value = constant->Copy();
          node.value = std::move(dataConstant);
        } else {
          // We're not a constant. Ew. (Maybe specialisation constant?)
          auto data = std::make_shared<ExistingValue>();
          data->id = constantId;
          node.value = std::move(data);
        }
      }
    }

    // Register the value returned by opVariable as pointing to that memory.
    PointerDetails pd;
    pd.type = context()->get_type_mgr()->GetRegisteredType(type);
    auto ptrInfo = std::make_shared<DataPointerTo>();
    ptrInfo->address = address;
    pd.address = ptrInfo;

    stack_pointer_details[node.owning_variable_id] = std::move(pd);

    // Increment the address of our next allocation, but add between 1 and 64
    // bytes of padding, so we have a nice alignment, and fences.
    address += node.size_in_bytes;
    address = (address + 64) & ~63;

    memory_tree.push_back(std::move(node));
  });

  // Now fill in the details all the way down the tree - so array of struct of
  // some vectors of floats ends up being floats sequentially laid out in memory
  // (assuming all the member offsets and array strides are correct).
  //
  // This also identifies all explicit pointer locations in memory, so we can
  // track their child expressions.
  for (auto& node : memory_tree) {
    recurse_populate_memory_tree(&node);
  }

  // Now trace all pointers. We have to repeat this process in case we have code
  // like (Excuse my mix of C++, glsl, and SpirV here):
  //
  //  size_t foo(size_t k) { return k + sizeof(float);}
  //
  //  void main()
  //  {
  //  Entry:
  //    float b[2] = {1.0, 0.5};
  //    size_t c;
  //    size_t d;
  //    float* e;
  //    Goto Work;
  //  Out:
  //    gl_FragColour = vec4(*e, b[0], b[1], b[0]);
  //    return;
  //  Work:
  //    c = reinterpret_cast<size_t>(b);
  //    d = foo(c);
  //    e = reinterpret_cast<float*>(c);
  //    goto Out;
  //  }
  //
  // We don't know that 'c' and d are really a pointer until we process the Work
  // block, and see that it gets the result of an OpPtrToU or OpBitCast
  // instruction, and we wont trace the output of foo until we see that k could
  // be a pointer, which we wont know until later.
  {
    bool foundAPointer = true;
    while (foundAPointer) {
      foundAPointer = false;

      context()->ProcessReachableCallTree(
          ProcessFunction([&](Function* F) -> bool {
            context()->cfg()->ForEachBlockInPostOrder(
                F->entry().get(), [&](BasicBlock* Block) {
                  foundAPointer |= find_and_track_pointers(Block);
                });
            return true;
          }));
    }
  }

  // Now step through the code, tracking all potential reads and writes, etc.
  uint32_t passCounter = 0;
  do {
    passCounter++;
    bool learntAnything = false;
    context()->ProcessReachableCallTree(
        ProcessFunction([&](Function* F) -> bool {
          std::list<BasicBlock*> blockQueue;

          context()->cfg()->ComputeStructuredOrder(F, F->entry().get(),
                                                   &blockQueue);

          for (auto block : blockQueue) {
            auto& preds = context()->cfg()->preds(block->id());
            if (preds.empty()) {
              learntAnything |= process(block, -1);
            }
            for (auto entry : preds) {
              learntAnything |= process(block, entry);
            }
          }

          return true;
        }));
    if (!learntAnything) break;
  } while (true);

  // Iterate over our state, and try to simplify out as much branching cases in
  // the memory map as possible.

  // Now look at temporary non-scalar non-heap object creation (eg
  // OpComposite<vec>, etc.) by tracing back through shuffles, inserts,
  // extracts, etc back to our now simplified memory access, to see if we can
  // forward what we've learnt from our memory analysis further out into the
  // code - beyond the OpLoad. Hopefully turning the string of loads,
  // BitCasts, Copies, shuffles, and inserts (typical result from an SSE / AVX /
  // etc optimisation engine) into a single clear glsl instruction. The goal is
  // to be able to initialise a vector or other composite type from a collection
  // of lvalues referencing directly into the structured blob the data came
  // from, without temporaries.

  // Now try to work out whether our analysis steps above removed the need for
  // some instructions entirely, and these could be skipped entirely from the
  // glsl export pass.

  std::unordered_set<uint64_t> rootAddressesNeeded;

  context()->ProcessReachableCallTree(ProcessFunction([&](Function* F) -> bool {
    F->ForEachInst([&](Instruction* I) {
      if (I->opcode() == SpvOpLoad) {
        auto ptr = I->GetSingleWordOperand(2);
        auto typeId = I->type_id();
        auto sizeOfType = size_of(*context()->get_type_mgr()->GetType(typeId));
        auto addresses = all_potential_stack_pointer_destinations(ptr);

        if (addresses.empty()) {
          assert(0 && " Warning - we can't do anything here. Plz add hints!");
        }

        for (auto& address : addresses) {
          auto data = id_data[I->result_id()];

          auto resolution = data->Resolved();
          if (resolution == Data::NotPossible ||
              resolution == Data::Undefined ||
              resolution == Data::NotPossibleWithoutHints) {
            auto root = this->find_memory_root_allocation(address);
            rootAddressesNeeded.insert(root->address);
          }
        }
      }

      if (I->opcode() == SpvOpCopyMemory ||
          I->opcode() == SpvOpCopyMemorySized) {
        auto targetPtr = I->GetSingleWordOperand(0);
        auto sourcePtr = I->GetSingleWordOperand(1);
        auto typeId = I->type_id();
        auto sizeOfType = size_of(*context()->get_type_mgr()->GetType(typeId));

        auto sourceAddresses =
            all_potential_stack_pointer_destinations(sourcePtr);
        auto targetAddresses =
            all_potential_stack_pointer_destinations(targetPtr);

        if (sourceAddresses.empty() || targetAddresses.empty()) {
          assert(0 && " Warning - we can't do anything here. Plz add hints!");
        }

        for (auto& address : sourceAddresses) {
          auto allocation = find_memory_location(address, sizeOfType);

          auto resolution = allocation->value->Resolved();
          if (resolution == Data::NotPossible ||
              resolution == Data::NotPossibleWithoutHints) {
            auto root = this->find_memory_root_allocation(address);
            rootAddressesNeeded.insert(root->address);
          }
        }
      }
    });

    return true;
  }));

  for (auto& tree : memory_tree) {
    if ((tree.storage_class == SpvStorageClassFunction ||
         tree.storage_class == SpvStorageClassPrivate) &&
        rootAddressesNeeded.find(tree.address) == rootAddressesNeeded.end()) {
      variables_to_cull.insert(tree.owning_variable_id);
    }
  }

  // So now we can freely erase any function / private variable access that
  // isn't within a rootVariablesNeeded ancestor. as well as any operation that
  // depends on it, making sure to replace the OpLoad(s) with the data we
  // discovered for it.
  std::unordered_set<uint32_t> redundantPointers;
  for (auto& pointer : stack_pointer_details) {
    auto addresses = all_potential_stack_pointer_destinations(pointer.first);

    bool allRedundant = true;
    for (auto& address : addresses) {
      auto* rootAllocation = find_memory_root_allocation(address);

      if (rootAllocation->storage_class != SpvStorageClassFunction &&
          rootAllocation->storage_class != SpvStorageClassPrivate) {
        // Someone else can set this memory at any time.
        allRedundant = false;
        break;
      }

      if (rootAddressesNeeded.find(rootAllocation->address) !=
          rootAddressesNeeded.end()) {
        // We need this allocation.
        allRedundant = false;
        break;
      }
    }

    if (allRedundant) redundantPointers.insert(pointer.first);
  }

  context()->ProcessReachableCallTree(ProcessFunction([&](Function* F) -> bool {
    F->ForEachInst([&](Instruction* I) {
      bool remove = false;
      if (I->HasResultId()) {
        auto returnValue = I->result_id();
        if (redundantPointers.find(returnValue) != redundantPointers.end()) {
          // We're returning a redundant pointer - remove.
          remove = true;
        }
      }

      for (auto i = 0u; i < I->NumInOperands(); i++) {
        if (I->GetInOperand(i).type == SPV_OPERAND_TYPE_ID) {
          if (redundantPointers.find(I->GetSingleWordInOperand(i)) !=
              redundantPointers.end()) {
            // We're using a redundant pointer - remove.
            remove = true;
          }
        }
      }

      if (remove) {
        if (I->opcode() == SpvOpLoad) {
          // We're loading from memory identified as removable - so we need to
          // replace the load with something a bit more useful.

          auto& value = id_data[I->result_id()];

          auto newSpirV = value->ConvertToSpirv(context(), this);

          auto outputId = newSpirV.back().back().result_id();

          I->SetOpcode(SpvOpCopyObject);
          I->SetInOperand(0, {outputId});

          if (newSpirV.size() == 1) {
            // We don't need to change the control flow graph - phew. Just
            // splice in the code.

            splice_before(std::move(newSpirV.front()), I);
          } else {
            assert(0 && "NYI");
          }
        } else if (I->opcode() == SpvOpCopyMemory ||
                   I->opcode() == SpvOpCopyMemorySized) {
          // We're copying either from or to removable memory.

          auto targetPtr = I->GetSingleWordOperand(0);
          auto sourcePtr = I->GetSingleWordOperand(1);

          bool targetIsRedundant =
              redundantPointers.find(targetPtr) != redundantPointers.end();
          bool sourceIsRedundant =
              redundantPointers.find(sourcePtr) != redundantPointers.end();

          if (targetIsRedundant && sourceIsRedundant) {
            // This memcopy should already be processed as having the target set
            // to the source.
          } else if (targetIsRedundant) {
            assert(0);  // I don't think this is possible - as it should've been
                        // mapped to external memory,
          } else if (sourceIsRedundant) {
            // We're reading from redundant memory, but writing to somewhere
            // that can't be simplified out. Eg memcpy(&gl_Position, &vec4(1.0),
            // sizeof(vec4)); This is now a store.
            I->SetOpcode(SpvOpStore);

            auto addresses =
                all_potential_stack_pointer_destinations(sourcePtr);

            auto destType = context()
                                ->get_type_mgr()
                                ->GetType(targetPtr)
                                ->AsPointer()
                                ->pointee_type();

            std::vector<int64_t> sizes = {size_of(*destType)};
            if (I->opcode() == SpvOpCopyMemorySized) {
              sizes = all_potential_int_values(I->GetSingleWordOperand(2));
            }

            if (addresses.size() == 0 || sizes.size() == 0) {
              assert(0 && "help - need hints");
            } else if (addresses.size() > 1 || sizes.size() > 1) {
              assert(0 && "NYI - need to modify CFG");
            } else {
              auto valueToStoreUnCasted = try_aligned_uncasted_memory_access(
                  *destType, addresses.front(), uint32_t(sizes.front()));

              if (!valueToStoreUnCasted) {
                // Need casting / compositing.
                auto valueToStore = try_aligned_access_memory(
                    *destType, addresses.front(), uint32_t(sizes.front()));
                assert(0 && "NYI");
              } else {
                // Can just use a simple value.
                auto newSpirV = valueToStoreUnCasted->value->ConvertToSpirv(
                    context(), this);

                I->ReplaceOperands(
                    {Operand(SPV_OPERAND_TYPE_ID, {targetPtr}),
                     Operand(SPV_OPERAND_TYPE_ID,
                             {newSpirV.back().back().result_id()})});

                splice_before(std::move(newSpirV.front()), I);
              }
            }
          }
        } else {
          // We're doing something else with a pointer to memory that we don't
          // care about. Just drop the instruction.
          I->RemoveFromList();
        }
      }
    });
    return true;
  }));

  // Now remove and forward all "OpCopyObject" - as we've abused it as a
  // forwarding mechanism within spir-v and it looks kinda messy.
  while (true) {
    context()->InvalidateAnalyses(IRContext::Analysis(-1));

    std::map<uint32_t, uint32_t> replacements;

    context()->ProcessReachableCallTree(
        ProcessFunction([&](Function* F) -> bool {
          F->ForEachInst([&](Instruction* I) {
            if (I->opcode() == SpvOpCopyObject) {
              auto generater = context()->get_def_use_mgr()->GetDef(
                  I->GetSingleWordInOperand(0));

              if (generater->opcode() != SpvOpCopyObject) {
                replacements[I->result_id()] = I->GetSingleWordInOperand(0);
              }

              I->RemoveFromList();
            }
          });
          return true;
        }));

    if (replacements.empty()) break;

    for (auto replace : replacements) {
      context()->ReplaceAllUsesWith(replace.first, replace.second);
    }
  }

  Instruction* memory_model = get_module()->GetMemoryModel();
  memory_model->SetInOperand(0, {SpvAddressingModelLogical});

  return Status::SuccessWithChange;
}

std::pair<uint32_t, uint32_t>
MemoryModelToLogical::access_chain_to_byte_offset_and_size(
    const analysis::Type& Type, uint32_t ThisStep,
    Instruction::const_iterator NextStep,
    Instruction::const_iterator EndOfSteps) {
  if (NextStep == EndOfSteps) {
    auto s = size_of(Type);
    return {uint32_t(s * ThisStep), s};
  }

  uint32_t nextStep = 0;

  auto tryGetNextConstant =
      context()->get_constant_mgr()->FindDeclaredConstant(NextStep->words[0]);

  if (tryGetNextConstant) {
    nextStep = tryGetNextConstant->AsIntConstant()->GetU32();
  } else {
    auto values = all_potential_int_values(NextStep->words[0]);
    if (values.size() == 1) {
      nextStep = uint32_t(values.front());
    } else {
      assert(0 && "NYI - we need to return a non-constant, stridded, offset.");
    }
  }

  if (Type.AsPointer()) {
    auto pointedType = Type.AsPointer()->pointee_type();
    auto sizeOfPointed = size_of(*pointedType);

    auto out = access_chain_to_byte_offset_and_size(*pointedType, nextStep,
                                                    NextStep + 1, EndOfSteps);
    out.first += ThisStep * sizeOfPointed;
    return out;
  } else if (Type.AsArray()) {
    auto arrayElement = Type.AsArray()->element_type();
    auto sizeOfElement = size_of(*arrayElement);

    auto arraySize = size_of(Type);
    auto stride = arraySize / child_types(Type).size();

    auto out = access_chain_to_byte_offset_and_size(*arrayElement, nextStep,
                                                    NextStep + 1, EndOfSteps);
    out.first += uint32_t(ThisStep * stride);
    return out;
  } else if (Type.AsStruct()) {
    auto childType = Type.AsStruct()->element_types()[ThisStep];

    auto elementEntry = Type.AsStruct()->element_decorations().find(ThisStep);

    uint32_t offset = -1;

    for (auto& deco : elementEntry->second) {
      if (deco.front() == SpvDecorationOffset) {
        offset = deco.back();
      }
    }

    assert(offset != -1);

    auto out = access_chain_to_byte_offset_and_size(*childType, nextStep,
                                                    NextStep + 1, EndOfSteps);

    out.first += offset;

    return out;
  } else if (Type.AsMatrix()) {
    // We're a matrix

    //:TODO: Figure out row major / column major stuff.
    assert(0 && "nyi");

    return {};
  } else if (Type.AsVector()) {
    // We're a vector.

    auto childElement = Type.AsVector()->element_type();
    auto sizeOfElement = size_of(*childElement);

    auto out = access_chain_to_byte_offset_and_size(*childElement, nextStep,
                                                    NextStep + 1, EndOfSteps);
    out.first += ThisStep * sizeOfElement;

    return out;
  } else {
    assert(0 && "Stepping into a scalar");
  }

  assert(0);
  return {};
}

bool MemoryModelToLogical::simulate_memcopy(uint64_t SourceAddress,
                                            uint64_t DestinationAddress,
                                            uint32_t SizeInBytes) {
  auto offset = 0u;

  while (offset < SizeInBytes) {
    auto sourceNode = find_memory_leaf_allocation(SourceAddress + offset);
    auto destNode = find_memory_leaf_allocation(DestinationAddress + offset);

    if (sourceNode->address != SourceAddress + offset) {
      assert(0 && "Misaligned read");
    }

    if (destNode->address != SourceAddress + offset) {
      assert(0 && "Misaligned write");
    }

    if (sourceNode->type != destNode->type) {
      assert(0 && "NYI - casting mem copy");
    }

    if (sourceNode->size_in_bytes != destNode->size_in_bytes) {
      assert(0 && "NYI - subdivide memory tree");
    }

    if (destNode->value) {
      // We're writing over memory. (We can't destNode->value =
      // sourceNode->value as the old value may have been read elsewhere)
      assert(0 && "NYI - phi memory trees");
    } else {
      // We're the first time a value has been written here.
      destNode->value = sourceNode->value;
    }

    offset += uint32_t(sourceNode->size_in_bytes);
  }

  return false;
}

bool MemoryModelToLogical::simulate_store(uint32_t SourceId,
                                          uint64_t DestinationAddress,
                                          uint32_t SizeInBytes) {
  auto& data = id_data[SourceId];

  if (!data) {
    // We don't know what we're storing.
    auto tryConstant =
        context()->get_constant_mgr()->GetConstantsFromIds({SourceId});

    if (tryConstant.empty()) {
      auto dataV = std::make_shared<ExistingValue>();
      dataV->id = SourceId;
      data = dataV;
    } else {
      auto dataV = std::make_shared<DataConstant>();
      dataV->value = tryConstant.front()->Copy();
      data = dataV;
    }
  }

  auto destNode = find_memory_location(DestinationAddress, SizeInBytes);

  if (destNode->address != DestinationAddress) {
    assert(0 && "Misaligned write handler NYI");
    // data = MisAlignedHandler(data)
  }

  if (destNode->value) {
    // We're writing over memory. (We can't destNode->value = sourceNode->value
    // as the old value may have been written elsewhere)
    assert(0 && "NYI - phi memory trees ");
  } else {
    // We're the first time a value has been written here.
    destNode->value = data;
  }

  if (destNode->has_child) {
    // We've written to a level in memory which has members, we need to update
    // all the children to refer to the new value.

    for (auto& child : destNode->children) {
      auto childData = std::make_shared<DataConstAccessChain>();
      childData->value = data;
      childData->steps = {child.local_access_chain.begin() +
                              destNode->local_access_chain.size(),
                          child.local_access_chain.end()};

      childData->offset = int64_t(child.address - destNode->address);

      if (child.value) {
        // We're writing over memory.
        assert(0 && "NYI - phi memory trees ");
      } else {
        // We're the first time a value has been written here.
        child.value = childData;
      }
    }
  }

  return false;
}

inline void MemoryModelToLogical::splice_before(InstructionList&& Instructions,
                                                Instruction* Destination) {
  while (!Instructions.empty()) {
    Instructions.front().InsertBefore(Destination);
  }
}

inline analysis::Integer* MemoryModelToLogical::GetUint32Type() {
  auto uint32t = context()->get_type_mgr()->GetRegisteredType(
      &analysis::Integer(32, false));

  if (uint32t == nullptr) {
    context()->get_type_mgr()->RegisterType(context()->TakeNextId(),
                                            analysis::Integer(32, false));
    uint32t = context()->get_type_mgr()->GetRegisteredType(
        &analysis::Integer(32, false));
  }

  return uint32t->AsInteger();
}

inline bool MemoryModelToLogical::DataBinaryOp::HasPtr() const {
  // I think all the cases where using a pointer in C++ arithmatic
  // this is valid.
  return (left->HasPtr() != right->HasPtr());
}

inline std::vector<uint64_t> MemoryModelToLogical::DataBinaryOp::PtrValues()
    const {
  std::vector<uint64_t> ptrs;
  std::vector<int64_t> ints;
  if (left->HasPtr()) {
    ptrs = left->PtrValues();
    ints = right->IntValues();
  } else {
    ptrs = right->PtrValues();
    ints = left->IntValues();
  }

  std::vector<uint64_t> output;
  for (auto p : ptrs) {
    for (auto i : ints) {
      if (opcode == SpvOpIAdd) {
        output.push_back(uint64_t(p + i));
      } else if (opcode == SpvOpISub) {
        output.push_back(uint64_t(p - i));
      } else if (opcode == SpvOpBitwiseAnd) {
        output.push_back(p & uint64_t(i));
      } else if (opcode == SpvOpBitwiseOr) {
        output.push_back(p | uint64_t(i));
      } else if (opcode == SpvOpBitwiseXor) {
        output.push_back(p ^ uint64_t(i));
      } else {
        assert(0 && "NYI");
      }
    }
  }
  return output;
}

inline std::vector<std::shared_ptr<MemoryModelToLogical::Data>>
MemoryModelToLogical::DataDynamicAccessChainLink::Dependancies() const {
  return {value, step};
}

inline MemoryModelToLogical::Data::ResolutionValueComplexity
MemoryModelToLogical::DataDynamicAccessChainLink::Resolved() const {
  return std::max(Data::Resolved(), SPIRVExpression);
}

inline bool MemoryModelToLogical::DataDynamicAccessChainLink::HasPtr() const {
  return true;
}

inline std::vector<uint64_t>
MemoryModelToLogical::DataDynamicAccessChainLink::PtrValues() const {
  auto ptrs = value->PtrValues();
  auto ints = step->IntValues();

  if (ptrs.empty() || ints.empty()) {
    assert(0 && "Help");
  }

  std::vector<uint64_t> out;
  for (auto ptr : ptrs) {
    for (auto i : ints) {
      out.push_back(ptr + i * stride);
    }
  }
  return out;
}

inline std::vector<std::shared_ptr<MemoryModelToLogical::Data>>
MemoryModelToLogical::DataConstAccessChain::Dependancies() const {
  return {value};
}

inline MemoryModelToLogical::Data::ResolutionValueComplexity
MemoryModelToLogical::DataConstAccessChain::Resolved() const {
  return std::max(Data::Resolved(), SPIRVExpression);
}

inline bool MemoryModelToLogical::DataConstAccessChain::HasPtr() const {
  return true;
}

inline std::vector<uint64_t>
MemoryModelToLogical::DataConstAccessChain::PtrValues() const {
  auto ptrs = value->PtrValues();

  if (ptrs.empty()) {
    assert(0 && "Help");
  }

  std::vector<uint64_t> out;
  for (auto ptr : ptrs) {
    out.push_back(ptr + offset);
  }
  return out;
}

inline std::vector<int64_t> MemoryModelToLogical::DataConstant::IntValues()
    const {
  auto intConstant = value->AsIntConstant();
  if (intConstant->type()->AsInteger()->IsSigned()) {
    if (intConstant->words().size() == 1) {
      return {intConstant->GetS32BitValue()};
    } else {
      return {intConstant->GetS64BitValue()};
    }
  } else {
    if (intConstant->words().size() == 1) {
      return {intConstant->GetU32BitValue()};
    } else {
      //:TODO (Ashley-Maptek) Ensure this still works when it comes to
      // arithmetic
      // when value is > 2^63
      return {int64_t(intConstant->GetU64BitValue())};
    }
  }
}

inline std::vector<double> MemoryModelToLogical::DataConstant::FloatValues()
    const {
  auto floatConstant = value->AsFloatConstant();
  if (floatConstant->words().size() == 2) {
    return {floatConstant->GetDouble()};
  } else {
    return {floatConstant->GetFloat()};
  }
  return {};
}

inline std::list<InstructionList>
MemoryModelToLogical::DataConstant::ConvertToSpirv(
    IRContext* Context, MemoryModelToLogical* Converter) {
  auto instruction =
      Context->get_constant_mgr()->GetDefiningInstruction(value.get());

  if (!instruction) {
    auto iterator = Context->types_values_end();
    instruction = Context->get_constant_mgr()->BuildInstructionAndAddToModule(
        value.get(), &iterator);
  }

  InstructionList block;
  Instruction copyInstruction(Context, SpvOpCopyObject,
                              Context->get_type_mgr()->GetId(value->type()),
                              Context->TakeNextId(), {});

  copyInstruction.AddOperand(
      Operand(SPV_OPERAND_TYPE_TYPE_ID, {instruction->result_id()}));

  block.push_back(std::make_unique<Instruction>(std::move(copyInstruction)));

  std::list<InstructionList> out;
  out.push_back(std::move(block));
  return out;
}

inline std::vector<double> MemoryModelToLogical::DataValueCast::FloatValues()
    const {
  //:TODO (Ashley-Maptek) The intricacies of the cast like dropping
  // bits of mantissa, etc, aren't implemented.
  if (value->HasFloat()) {
    return value->FloatValues();
  } else if (value->HasInt()) {
    std::vector<double> out;
    std::vector<int64_t> in = value->IntValues();
    out.reserve(in.size());
    for (auto i : in) {
      out.push_back(double(i));
    }
    return out;
  }
  assert(false);
  return {};
}

inline std::vector<int64_t> MemoryModelToLogical::DataValueCast::IntValues()
    const {
  //:TODO (Ashley-Maptek) The intricacies of the cast like modulating / sign
  // shifting, etc, aren't implemented yet. So uint8(257) == 1, but
  // uint32(uint8(257)) still equals 257, and float(int(3.14)) == 3.14.

  if (value->HasInt()) {
    return value->IntValues();
  } else if (value->HasFloat()) {
    std::vector<int64_t> out;
    std::vector<double> in = value->FloatValues();
    out.reserve(in.size());
    for (auto i : in) {
      out.push_back(int64_t(i));
    }
    return out;
  }
  assert(false);
  return {};
}

inline std::list<InstructionList>
MemoryModelToLogical::DataComposition::ConvertToSpirv(
    IRContext* Context, MemoryModelToLogical* Converter) {
  InstructionList entryBlock;
  std::list<InstructionList> branchingCode;
  std::vector<uint32_t> types;
  std::vector<uint32_t> ids;

  for (auto& value : values) {
    auto result = value->ConvertToSpirv(Context, Converter);

    ids.push_back(result.back().back().result_id());
    types.push_back(result.back().back().type_id());

    if (result.size() == 1) {
      entryBlock.Splice(entryBlock.end(), &result.front(),
                        result.front().begin(), result.front().end());
    } else {
      branchingCode.insert(branchingCode.end(),
                           std::make_move_iterator(result.begin()),
                           std::make_move_iterator(result.end()));
    }
  }

  auto joiner = std::make_unique<Instruction>(
      Context, SpvOpCompositeConstruct,
      Context->get_type_mgr()->GetId(output_type), Context->TakeNextId(),
      Instruction::OperandList());

  for (auto value : ids) {
    joiner->AddOperand(Operand(SPV_OPERAND_TYPE_ID, {value}));
  }

  if (branchingCode.size()) {
    branchingCode.front().Splice(branchingCode.front().begin(), &entryBlock,
                                 entryBlock.begin(), entryBlock.end());

    branchingCode.back().push_back(std::move(joiner));

    return branchingCode;
  }

  entryBlock.push_back(std::move(joiner));

  std::list<InstructionList> output;
  output.push_back(std::move(entryBlock));
  return output;
}

inline std::vector<std::shared_ptr<MemoryModelToLogical::Data>>
MemoryModelToLogical::Data::Dependancies() const {
  return {};
}

inline MemoryModelToLogical::Data::ResolutionValueComplexity
MemoryModelToLogical::Data::Resolved() const {
  ResolutionValueComplexity r = Undefined;
  for (auto d : Dependancies()) {
    auto v = d->Resolved();
    r = std::max(r, v);
  }
  return r;
}

inline MemoryModelToLogical::Data::ResolutionCfgComplexity
MemoryModelToLogical::Data::ResolvedCfgComplexity() const {
  ResolutionCfgComplexity r = NoResolution;
  for (auto d : Dependancies()) {
    auto v = d->ResolvedCfgComplexity();
    r = std::max(r, v);
  }
  return r;
}

inline std::list<InstructionList> MemoryModelToLogical::Data::ConvertToSpirv(
    IRContext* Context, MemoryModelToLogical* Converter) {
  assert(0 && "NYI");
  return std::list<InstructionList>();  // Note return {} needs copyable
                                        // type.
}

}  // namespace opt
}  // namespace spvtools